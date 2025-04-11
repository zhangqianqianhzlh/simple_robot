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
from PIL import Image, ImageDraw, ImageFont
import tempfile

def convert_image_to_base64(image):
    """Convert an image to base64 format."""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def send_api_request(image_base64, api_url):
    """Send a request to the action proposal API."""
    payload = {
        "image": image_base64,
        "min_angle": 20,
        "number_size": 30,
        "min_path_length": 100
    }
    response = requests.post(api_url, json=payload)
    return response

def handle_api_response(response):
    """Handle the API response and return the augmented image and actions."""
    if response.status_code == 200:
        data = response.json()
        augmented_image_base64 = data["image"]
        augmented_image_bytes = base64.b64decode(augmented_image_base64)
        augmented_image_np = cv2.imdecode(np.frombuffer(augmented_image_bytes, np.uint8), cv2.IMREAD_COLOR)
        actions = data["actions"]
        return augmented_image_np, actions
    else:
        st.error(f"Error: API returned status code {response.status_code}")
        return None, None

def get_action_proposals(image, api_url="http://10.8.25.28:8075/generate_action_proposals"):
    """Get action proposals from the API."""
    image_base64 = convert_image_to_base64(image)
    response = send_api_request(image_base64, api_url)
    return handle_api_response(response)

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

def setup_views_directory():
    """Ensure the views directory is ready for storing images."""
    if not os.path.exists('views'):
        os.makedirs('views')
    else:
        shutil.rmtree('views')
        os.makedirs('views')

def parse_vlm_output(vlm_output_str):
    """Parse the VLM output to extract reasoning and action."""
    reasoning = "Could not parse VLM output."
    action_chosen = None
    try:
        json_pattern = re.compile(r"```json\s*({.*?})\s*```|({.*?})", re.DOTALL)
        match = json_pattern.search(vlm_output_str)
        if match:
            json_str = match.group(1) or match.group(2)
            if json_str:
                vlm_output_json = json.loads(json_str)
                reasoning = vlm_output_json.get("reasoning", "No reasoning provided in JSON.")
                action_chosen = vlm_output_json.get("action", None)
        else:
            vlm_output_json = json.loads(vlm_output_str.strip())
            reasoning = vlm_output_json.get("reasoning", "No reasoning provided.")
            action_chosen = vlm_output_json.get("action", None)
    except json.JSONDecodeError:
        st.error("Failed to parse VLM output as JSON.")
    return reasoning, action_chosen

def execute_action(env, action_number, actions_info):
    """
    Execute the chosen action by rotating to the appropriate degree and moving forward
    """
    # Find the action info for the chosen action number
    action_info = next((a for a in actions_info if a["action_number"] == action_number), None)
    
    if action_info is None:
        st.error(f"Error: Action {action_number} not found in action proposals")
        return None
    
    degree = action_info["turning_degree"]
    st.write(f"Executing action {action_number} with degree {degree}")
    
    # If action is 0, turn around 180 degrees
    if action_number == 0:
        event = env.step("RotateRight", degrees=180)
        return event
    
    # Otherwise, rotate to the specific degree and move forward
    # First determine if we need to rotate left or right
    if degree < 0:
        # Negative degree means turn left
        rotation_action = "RotateLeft"
    else:
        # Positive degree means turn right
        rotation_action = "RotateRight"
    
    # Execute the rotation
    event = env.step(rotation_action, degrees=abs(degree))
    
    # Move forward
    event = env.step("MoveAhead")
    return event

def cv2_to_streamlit_image(image):
    """Convert OpenCV image to format suitable for Streamlit display"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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

    # Sidebar for configuration
    st.sidebar.header("Simulation Settings")
    
    # Floor plan selection
    floor_id = st.sidebar.text_input(
        "Select Floor Plan", 
        value="FloorPlan10"
    )
    
    # Model selection
    model_id = st.sidebar.text_input(
        "Model ID", 
        value="Pro/Qwen/Qwen2.5-VL-7B-Instruct"
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
        value="find the white fridge"
    )
    
    # Task prompt template
    task_prompt = st.sidebar.text_area(
        "Task Prompt", 
        value="""You are a robot navigating a house. You see an image with numbered paths representing possible directions.
Your task is to follow these instructions: '{TARGET}' and get close to it.
Analyze the current view and the available paths ({ACTIONS}).
Provide a very short reasoning steps and then choose the best path number.
If you believe the task is complete based on the view, choose 'Done'.
If no path seems helpful, choose '0' to turn around.
Output your response as a JSON object with two keys: "reasoning" (your reasoning) and "action" (the chosen number as a string, 'Done', or '0').
Example: {{"reasoning": "The target is the fridge. Path 3 seems to lead towards the kitchen area where fridges usually are.", "action": "3"}}
Example: {{"reasoning": "The target fridge is clearly visible right in front of me.", "action": "Done"}}"""
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
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'vlm_output_str' not in st.session_state:
        st.session_state.vlm_output_str = "No output available."

    # Main display area
    main_container = st.container()
    with st.spinner("Initializing simulation..."):
        # Create the environment first
        env = ThorEnvDogView(floor_id)

    # Start simulation
    if start_button:
        with st.spinner("Initializing simulation..."):
            # Reset the environment
            env.reset(floor_id)
            # Create the agent with the environment
            st.session_state.agent = VLMNavigationAgent(env, model_id, api_url, max_distance_to_move)
            st.session_state.agent.reset()
            st.session_state.running = True
            main_container.header("Simulation Started")
            
            # Display initial view and depth
            col1, col2 = main_container.columns(2)
            with col1:
                main_container.image(cv2_to_streamlit_image(st.session_state.agent.view), caption="Initial View", use_container_width=True)
            with col2:
                depth_heatmap = depth_to_heatmap(st.session_state.agent.depth)
                main_container.image(cv2_to_streamlit_image(depth_heatmap), caption="Depth Map", use_container_width=True)

    # Stop simulation
    if stop_button:
        st.session_state.running = False
        main_container.warning("Simulation stopped by user")

    # Run simulation steps
    if st.session_state.running and st.session_state.agent is not None and not st.session_state.agent.completed:
        # Continue until max steps or stopped
        if st.session_state.agent.step_number < max_steps:
            # Create a new container for each step
            step_container = main_container.container()
            step_container.subheader(f"Step {st.session_state.agent.step_number + 1}")

            # Set up columns for augmented view and model's output
            col1, col2 = step_container.columns(2)

            try:
                # Execute one step of the navigation
                augmented_view, actions_info, reasoning, action_chosen, new_view = st.session_state.agent.step(
                    target, task_prompt, max_steps
                )

                if augmented_view is None:  # Simulation completed or max steps reached
                    st.session_state.running = False
                    return

                # Store the VLM output string
                st.session_state.vlm_output_str = st.session_state.agent.vlm_output_str

                # Draw green circle at the center position of the chosen action
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

                # Display augmented view and depth
                with col1:
                    step_container.image(cv2_to_streamlit_image(augmented_view), caption="Augmented View with Action Paths", width=300)
                    if st.session_state.agent.depth is not None:
                        depth_heatmap = depth_to_heatmap(st.session_state.agent.depth)
                        step_container.image(cv2_to_streamlit_image(depth_heatmap), caption="Depth Map", width=300)

                # Display model's raw output and choice of action
                with col2:
                    step_container.text("VLM Raw Output:")
                    step_container.code(st.session_state.vlm_output_str, language=None)
                    step_container.markdown("**VLM Reasoning (from JSON):**")
                    step_container.info(reasoning)
                    step_container.text(f"VLM Chose Action (from JSON): {action_chosen}")

                if new_view is not None:
                    with col2:
                        step_container.image(cv2_to_streamlit_image(new_view), caption=f"After Action {action_chosen}", use_container_width=True)
                else:
                    step_container.warning("No new view available - action may have failed")

                time.sleep(step_delay)
                st.rerun()

            except Exception as e:
                step_container.error(f"Error during simulation step: {str(e)}")
                st.session_state.running = False
                return

        else:
            st.session_state.running = False
            st.session_state.agent.completed = True
            main_container.warning("Maximum steps reached.")
    
    # Display action history with images
    if st.session_state.agent is not None and st.session_state.agent.step_number > 0:
        steps_per_page = 20
        total_steps = len(st.session_state.agent.action_memory)
        total_pages = (total_steps + steps_per_page - 1) // steps_per_page

        page_number = st.sidebar.number_input("Page", min_value=1, max_value=total_pages, value=1)
        start_index = (page_number - 1) * steps_per_page
        end_index = start_index + steps_per_page

        # Add video creation button
        if st.sidebar.button("Create Simulation Video"):
            with st.spinner("Creating video..."):
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_file:
                    video_path = create_simulation_video(st.session_state.agent, tmp_file.name)
                    if video_path:
                        with open(video_path, 'rb') as f:
                            video_bytes = f.read()
                            st.sidebar.download_button(
                                label="Download Simulation Video",
                                data=video_bytes,
                                file_name="simulation_video.mp4",
                                mime="video/mp4"
                            )
                        os.unlink(video_path)  # Clean up temporary file

        with st.expander("Action History"):
            for i, (action_str, image) in enumerate(zip(
                st.session_state.agent.action_memory[start_index:end_index],
                st.session_state.agent.augmented_images[start_index:end_index]
            ), start=start_index):
                # Create two columns for each step
                col1, col2 = st.columns(2)
                
                # Display the action text
                st.text(f"Step {i}: {action_str}")
                
                # Display augmented view and depth map side by side
                with col1:
                    st.image(cv2_to_streamlit_image(image), caption=f"Augmented View for Step {i}", width=300)
                with col2:
                    if hasattr(st.session_state.agent, 'depth_memory') and i < len(st.session_state.agent.depth_memory):
                        depth_heatmap = depth_to_heatmap(st.session_state.agent.depth_memory[i])
                        st.image(cv2_to_streamlit_image(depth_heatmap), caption=f"Depth Map for Step {i}", width=300)

    # Example of folding VLM raw output
    with st.expander("VLM Raw Output"):
        st.code(st.session_state.vlm_output_str, language=None)

if __name__ == "__main__":
    main() 