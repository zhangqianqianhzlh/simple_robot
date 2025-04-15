import streamlit as st
import cv2
import os
import shutil
import base64
import requests
import json
import numpy as np
import time
import re # Import the regex module
from models import VLM
from env import ThorEnvDogView
from agent import VLMNavigationAgent
from agent2 import VLMNavigationAgent2
from PIL import Image, ImageDraw, ImageFont
import tempfile
from functools import lru_cache
from datetime import datetime


def initialize_simulation(floor_id):
    """Initialize the simulation environment and model."""
    model = VLM("llm.yaml")
    env = ThorEnvDogView(floor_id)
    return model, env

def reset_simulation_state():
    """Reset the Streamlit session state for a new simulation."""
    st.session_state.running = True
    st.session_state.step_number = 0
    st.session_state.action_memory = []
    st.session_state.completed = False
    st.session_state.augmented_images = []  # Store augmented images




@st.cache_data
def cv2_to_streamlit_image(image):
    """Convert OpenCV image to format suitable for Streamlit display"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

@st.cache_data
def depth_to_heatmap(depth):
    """Convert depth array to heatmap visualization"""
    depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_normalized.astype(np.uint8)
    return cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)

def create_simulation_video(agent, output_path='simulation_video.mp4'):
    """Create a video from the simulation history."""
    if not agent or not agent.augmented_images:
        st.error("No simulation history available to create video")
        return None
    
    # Get video dimensions from the first image
    height, width = agent.augmented_images[0].shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, 1.0, (width * 2, height))  # Double width for side-by-side
    
    try:
        for i, (augmented_img, depth_img) in enumerate(zip(agent.augmented_images, agent.depth_memory)):
            # Convert depth to heatmap
            depth_heatmap = depth_to_heatmap(depth_img)
            
            # Convert BGR to RGB for PIL
            augmented_rgb = cv2.cvtColor(augmented_img, cv2.COLOR_BGR2RGB)
            depth_rgb = cv2.cvtColor(depth_heatmap, cv2.COLOR_BGR2RGB)
            
            # Create PIL images
            augmented_pil = Image.fromarray(augmented_rgb)
            depth_pil = Image.fromarray(depth_rgb)
            
            # Create a new image with double width
            combined = Image.new('RGB', (width * 2, height))
            combined.paste(augmented_pil, (0, 0))
            combined.paste(depth_pil, (width, 0))
            
            # Add text overlay
            draw = ImageDraw.Draw(combined)
            try:
                font = ImageFont.truetype("arial.ttf", 30)
            except:
                font = ImageFont.load_default()
            
            # Add step number and action
            step_text = f"Step {i+1}"
            if i < len(agent.action_memory):
                action_text = f"Action: {agent.action_memory[i]}"
                draw.text((10, 10), step_text, (255, 255, 255), font=font)
                draw.text((10, 50), action_text, (255, 255, 255), font=font)
            
            # Convert back to OpenCV format
            combined_cv = cv2.cvtColor(np.array(combined), cv2.COLOR_RGB2BGR)
            video.write(combined_cv)
        
        video.release()
        return output_path
    except Exception as e:
        st.error(f"Error creating video: {str(e)}")
        if video:
            video.release()
        return None

def main():
    st.set_page_config(page_title="Robot Navigation Simulation", layout="wide")
    st.title("VLM Navigation Simulation")

    # 创建输出目录（如果不存在）
    if not os.path.exists("output"):
        os.makedirs("output")
        
    # 保存当前时间戳到session state（如果不存在）
    if 'current_run_id' not in st.session_state:
        st.session_state.current_run_id = None
    
    # Sidebar for configuration
    st.sidebar.header("Simulation Settings")
    
    # Floor plan selection
    floor_id = st.sidebar.text_input(
        "Select Floor Plan", 
        value="FloorPlan_Train1_5"
    )
    
    # Model selection - 改为下拉选择框
    model_options = ["qwen2.5-vl-7b-instruct", "qwen2.5-vl-32b-instruct", "Qwen/Qwen2.5-VL-72B-Instruct"]
    model_id = st.sidebar.selectbox(
        "选择模型", 
        options=model_options,
        index=0
    )
    
    # API URL
    api_url = st.sidebar.text_input(
        "Action Proposal API URL", 
        value="http://10.8.25.28:8075/generate_action_proposals"
    )
    
    # Max distance to move
    max_distance_to_move = st.sidebar.number_input(
        "Maximum Distance to Move (meters)",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1
    )
    
    # Target description
    target = st.sidebar.text_area(
        "Navigation Target", 
        value="find the TV"
    )
    
    # Task prompt template
    task_prompt = st.sidebar.text_area(
        "Task Prompt", 
        value="""You are a robot navigating a house. You see an image with numbered paths representing possible directions.
Your task is to follow these navigation instructions: '{TARGET}'
First, briefly describe what you see in the current view (e.g., "I see a kitchen with a counter and cabinets").
Then analyze the available paths ({ACTIONS}) and choose the best path number to follow the instructions.
If no path seems helpful, choose '0' to turn around.
Output your response as a JSON object with two keys: "reasoning" (your description and reasoning) and "action" (the chosen number as a string or '0').
Example: {{"reasoning": "I see a kitchen with a counter and cabinets. The instructions say to go left and then find the fridge. Path 3 leads to the left, which matches the first part of the instructions.", "action": "3"}}
Example: {{"reasoning": "I see a dead end with no clear paths forward. I should turn around to explore other directions.", "action": "0"}}"""
    )
    
    # Step delay
    step_delay = st.sidebar.slider(
        "Delay between steps (seconds)", 
        min_value=0.0, 
        max_value=5.0, 
        value=0.0, 
        step=0.1
    )
    
    # Maximum steps
    max_steps = st.sidebar.number_input(
        "Maximum Steps", 
        min_value=1, 
        max_value=100, 
        value=50
    )

    # Control buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_button = st.button("Start Simulation")
    with col2:
        stop_button = st.button("Stop Simulation")

    # Initialize session state for tracking simulation
    if 'step_history' not in st.session_state:
        st.session_state.step_history = []
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'vlm_output_str' not in st.session_state:
        st.session_state.vlm_output_str = "No output available."
    if 'last_step_time' not in st.session_state:
        st.session_state.last_step_time = time.time()
    if 'initial_view' not in st.session_state:
        st.session_state.initial_view = None
    if 'initial_depth' not in st.session_state:
        st.session_state.initial_depth = None

    # 使用单一容器来控制所有输出的顺序
    main_container = st.container()
    
    # 首先创建一个用于显示初始视图的容器
    initial_view_container = st.container()
    
    # 然后创建一个用于显示步骤的容器
    steps_container = st.container()

    # Start simulation
    if start_button:
        with st.spinner("Initializing simulation..."):
            # 创建以时间命名的新文件夹
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_folder = os.path.join("output", timestamp)
            os.makedirs(run_folder, exist_ok=True)
            st.session_state.current_run_id = timestamp
            
            # 保存配置信息到文件
            config_info = {
                "timestamp": timestamp,
                "floor_id": floor_id,
                "model_id": model_id,
                "api_url": api_url,
                "max_distance": max_distance_to_move,
                "target": target,
                "max_steps": max_steps
            }
            with open(os.path.join(run_folder, "config.json"), "w") as f:
                json.dump(config_info, f, indent=4)
            
            # Create the environment first
            env = ThorEnvDogView(floor_id)
            # Create the agent with the environment
            st.session_state.agent = VLMNavigationAgentNoMemory(env, model_id, api_url, max_distance_to_move)
            st.session_state.agent.reset()
            st.session_state.running = True
            
            # 保存初始视图到session_state和文件
            st.session_state.initial_view = st.session_state.agent.view.copy() if st.session_state.agent.view is not None else None
            st.session_state.initial_depth = st.session_state.agent.depth.copy() if st.session_state.agent.depth is not None else None
            
            # 保存初始视图
            if st.session_state.initial_view is not None:
                initial_view_path = os.path.join(run_folder, "initial_view.jpg")
                cv2.imwrite(initial_view_path, st.session_state.initial_view)
            
            if st.session_state.initial_depth is not None:
                initial_depth_path = os.path.join(run_folder, "initial_depth.jpg")
                depth_heatmap = depth_to_heatmap(st.session_state.initial_depth)
                cv2.imwrite(initial_depth_path, depth_heatmap)
            
            # 清空步骤历史
            st.session_state.step_history = []

    # 将这部分代码移动到 start_button 条件块之外，确保每次应用重新渲染时都能显示
    # 在main_container之后显示初始视图（如果存在）
    if st.session_state.initial_view is not None:
        with initial_view_container:
            st.header("Simulation Started")
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(cv2_to_streamlit_image(st.session_state.initial_view), caption="Initial View")
            with col2:
                if st.session_state.initial_depth is not None:
                    depth_heatmap = depth_to_heatmap(st.session_state.initial_depth)
                    st.image(cv2_to_streamlit_image(depth_heatmap), caption="Depth Map")

    # Stop simulation
    if stop_button:
        st.session_state.running = False
        with main_container:
            st.warning("Simulation stopped by user")

    # Run simulation steps
    if st.session_state.running and st.session_state.agent is not None:
        # Continue until max steps or stopped
        if st.session_state.agent.step_number < max_steps:
            # 在步骤容器内添加新步骤
            with steps_container:
                try:
                    # 执行导航的一个步骤
                    print(f"before memory: {st.session_state.agent.memory.location_history}")
                    augmented_view, actions_info, reasoning, action_chosen, new_view, is_completed = st.session_state.agent.step(
                        target, task_prompt, max_steps
                    )
                    print(f"after memory: {st.session_state.agent.memory.location_history}")
                    st.text(st.session_state.agent.location)

                    if augmented_view is None:  # 模拟结束或达到最大步骤
                        print(f" task ended at step {st.session_state.agent.step_number}")
                        st.session_state.running = False
                        
                        # 保存整个运行的摘要信息
                        if st.session_state.current_run_id:
                            run_folder = os.path.join("output", st.session_state.current_run_id)
                            summary = {
                                "total_steps": st.session_state.agent.step_number,
                                "completed": st.session_state.agent.completed,
                                "target": target,
                                "model": model_id,
                                "timestamp_end": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "success": is_completed
                            }
                            with open(os.path.join(run_folder, "summary.json"), "w") as f:
                                json.dump(summary, f, indent=4)
                        
                        if is_completed:
                            with main_container:
                                st.success("Task completed successfully!")
                        else:
                            with main_container:
                                st.error("Task failed!")
                        return

                    # 保存VLM输出字符串
                    st.session_state.vlm_output_str = st.session_state.agent.vlm_output_str

                    # 在所选动作的中心位置绘制绿色圆圈
                    if action_chosen is not None and action_chosen != 'done':
                        try:
                            action_number = int(action_chosen)
                            # Find the action info for the chosen action number
                            action_info = next((a for a in actions_info if a["action_number"] == action_number), None)
                            if action_info and "center_position" in action_info:
                                center_x, center_y = action_info["center_position"]
                                # Draw a larger green circle with outline (thickness=2) instead of fill
                                cv2.circle(augmented_view, (center_x, center_y), 20, (0, 255, 0), 2)
                                
                                # If there's a boundary point, draw it in red
                                if action_info["boundary_point"] is not None:
                                    boundary_x, boundary_y = action_info["boundary_point"]
                                    cv2.circle(augmented_view, (boundary_x, boundary_y), 10, (0, 0, 255), 2)
                        except ValueError:
                            pass  # Skip if action_chosen is not a number



                    # 创建当前步骤的数据
                    current_step = {
                        "step_number": st.session_state.agent.step_number,
                        "view": augmented_view.copy() if augmented_view is not None else None,
                        "depth": depth_heatmap.copy() if depth_heatmap is not None else None,
                        "vlm_output": st.session_state.vlm_output_str,  # 替换为实际输出
                        "reasoning": reasoning,   # 替换为实际推理
                        "action": action_chosen       # 替换为实际动作
                    }
                    
                    # 将当前步骤添加到历史记录中
                    st.session_state.step_history.append(current_step)
                    
                    # 保存当前步骤的数据到输出文件夹
                    if st.session_state.current_run_id:
                        run_folder = os.path.join("output", st.session_state.current_run_id)
                        step_prefix = f"step{st.session_state.agent.step_number}_"
                        
                        # 保存视图图像
                        if augmented_view is not None:
                            view_path = os.path.join(run_folder, f"{step_prefix}augmented_view.jpg")
                            cv2.imwrite(view_path, augmented_view)
                        
                        # 保存深度图像
                        if st.session_state.agent.depth is not None:
                            depth_path = os.path.join(run_folder, f"{step_prefix}depth.jpg")
                            depth_heatmap = depth_to_heatmap(st.session_state.agent.depth)
                            cv2.imwrite(depth_path, depth_heatmap)
                        
                        # 保存新视图（如果有）
                        if new_view is not None:
                            new_view_path = os.path.join(run_folder, f"{step_prefix}new_view.jpg")
                            cv2.imwrite(new_view_path, new_view)

                        #update the location map and save it
                        new_map = st.session_state.agent.memory.draw_map()
                        map_path = os.path.join(run_folder, f"{step_prefix}map.jpg")
                        cv2.imwrite(map_path, new_map)

                        
                        # 保存VLM输出和动作信息
                        step_info = {
                            "step_number": st.session_state.agent.step_number,
                            "vlm_output": st.session_state.vlm_output_str,
                            "reasoning": reasoning,
                            "action_chosen": action_chosen,
                            "actions_info": actions_info,
                            "is_completed": is_completed
                        }
                        with open(os.path.join(run_folder, f"{step_prefix}info.json"), "w") as f:
                            json.dump(step_info, f, indent=4, ensure_ascii=False)
                    
                    # 显示所有历史步骤
                    if len(st.session_state.step_history) > 0:
                        steps_per_page = 20
                        total_pages = (len(st.session_state.step_history) - 1) // steps_per_page + 1
                        
                        # 只有当总页数大于1时才显示滑块
                        if total_pages > 1:
                            current_page = st.sidebar.slider("页码", 1, total_pages, total_pages)  # 默认显示最新页
                        else:
                            current_page = 1
                        
                        start_idx = (current_page - 1) * steps_per_page
                        end_idx = min(start_idx + steps_per_page, len(st.session_state.step_history))
                        
                        # 只显示当前页的步骤
                        for step in st.session_state.step_history[start_idx:end_idx]:
                            st.subheader(f"Step {step['step_number']}")
                            


                            # 设置列用于显示
                            col1, col2 = st.columns(2)
                            
                            # 显示增强视图和深度图
                            with col1:
                                if step['view'] is not None:
                                    st.image(cv2_to_streamlit_image(step['view']), caption="Augmented View with Action Paths", width=300)
                                if step['depth'] is not None:
                                    depth_heatmap = depth_to_heatmap(step['depth'])
                                    st.image(cv2_to_streamlit_image(depth_heatmap), caption="Depth Map", width=300)

                            # 显示模型的原始输出和选择的动作
                            with col2:
                                st.text("VLM Raw Output:")
                                st.code(step['vlm_output'], language=None)
                                st.markdown("**VLM Reasoning (from JSON):**")
                                st.info(step['reasoning'])
                                st.text(f"VLM Chose Action (from JSON): {step['action']}")

                                # 显示任务完成检查信息（如果有）
                                if hasattr(st.session_state.agent.memory, 'completion_checks') and st.session_state.agent.memory.completion_checks:
                                    latest_check = st.session_state.agent.memory.completion_checks[-1]
                                    st.markdown("**Latest Task Completion Check:**")
                                    status = "✅ Completed" if latest_check['completed'] else "❌ Not Completed"
                                    st.markdown(f"Status: {status}")
                                    st.markdown(f"Reasoning: {latest_check['reasoning']}")


                                st.subheader(f"Step {step['step_number']} done ..")

                            if new_view is not None:
                                with col2:
                                    st.image(cv2_to_streamlit_image(new_view), caption=f"After Action {action_chosen}", use_container_width=True)
                            else:
                                with col2:
                                    st.warning("No new view available - action may have failed")
                    
                    # 计算自上一步以来的时间并在需要时休眠
                    current_time = time.time()
                    elapsed = current_time - st.session_state.last_step_time
                    if elapsed < step_delay:
                        time.sleep(step_delay - elapsed)
                    st.session_state.last_step_time = time.time()
                    
                    

                    # 只有在未达到最大步数时才重新运行
                    if st.session_state.agent.step_number < max_steps:
                        st.rerun()

                except Exception as e:
                    with steps_container:
                        st.error(f"Error during simulation step: {str(e)}")
                    st.session_state.running = False
                    return

        else:
            st.session_state.running = False
            st.session_state.agent.completed = True
            
            # 保存整个运行的摘要信息（达到最大步数的情况）
            if st.session_state.current_run_id:
                run_folder = os.path.join("output", st.session_state.current_run_id)
                summary = {
                    "total_steps": st.session_state.agent.step_number,
                    "completed": st.session_state.agent.completed,
                    "target": target,
                    "model": model_id,
                    "timestamp_end": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "reason": "Maximum steps reached"
                }
                with open(os.path.join(run_folder, "summary.json"), "w") as f:
                    json.dump(summary, f, indent=4)
            
            with main_container:
                st.warning("Maximum steps reached.")
    






if __name__ == "__main__":
    main() 