import streamlit as st
import cv2
import os
import numpy as np
from env import ThorEnvDogView
import shutil
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# 设置页面配置
st.set_page_config(
    page_title="AI2Thor 环境交互控制器",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 自定义CSS来增加最大宽度
st.markdown("""
<style>
    .reportview-container .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# 绘制平面图的函数
def draw_map_with_agent(event):
    """
    绘制环境平面图并标记机器人位置
    """
    print("正在生成平面图...")
    
    # 尝试获取环境边界
    try:
        # 获取环境边界
        scene_bounds = event.metadata['sceneBounds']
        print(f"场景边界数据结构类型: {type(scene_bounds)}")
        print(f"场景边界数据内容: {scene_bounds}")
        
        # 尝试不同的键获取边界信息
        if isinstance(scene_bounds, dict) and 'x' in scene_bounds and 'z' in scene_bounds:
            # 原始结构
            x_min = scene_bounds['x']['min']
            x_max = scene_bounds['x']['max']
            z_min = scene_bounds['z']['min']
            z_max = scene_bounds['z']['max']
            print(f"使用x/z边界: X({x_min}, {x_max}), Z({z_min}, {z_max})")
        elif isinstance(scene_bounds, dict) and 'cornerPoints' in scene_bounds:
            # 如果是cornerPoints结构
            points = scene_bounds['cornerPoints']
            x_coords = [p[0] for p in points]
            z_coords = [p[2] for p in points]  # 假设y是高度，z是深度
            x_min, x_max = min(x_coords), max(x_coords)
            z_min, z_max = min(z_coords), max(z_coords)
            print(f"使用cornerPoints边界: X({x_min}, {x_max}), Z({z_min}, {z_max})")
        elif isinstance(scene_bounds, dict) and 'center' in scene_bounds:
            # 如果是center和size结构
            center = scene_bounds['center']
            size = scene_bounds.get('size', {'x': 10, 'z': 10})  # 默认值
            x_min = center['x'] - size['x']/2
            x_max = center['x'] + size['x']/2
            z_min = center['z'] - size['z']/2
            z_max = center['z'] + size['z']/2
            print(f"使用center/size边界: X({x_min}, {x_max}), Z({z_min}, {z_max})")
        else:
            # 无法识别的结构，使用默认边界
            st.warning("无法识别的sceneBounds结构，使用默认边界")
            x_min, x_max = -5, 5
            z_min, z_max = -5, 5
    except Exception as e:
        st.error(f"处理sceneBounds时出错: {str(e)}")
        # 使用默认边界
        x_min, x_max = -5, 5
        z_min, z_max = -5, 5
    
    # 创建一个简单的平面图（白色背景）
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(z_min, z_max)
    ax.set_xlabel('X 坐标')
    ax.set_ylabel('Z 坐标')
    ax.set_title('环境平面图与机器人位置')
    ax.grid(True)
    
    # 获取机器人的位置
    agent_position = event.metadata['agent']['position']
    agent_rotation = event.metadata['agent']['rotation']
    
    # 在平面图上标记机器人的位置（红色圆点）
    circle = Circle((agent_position['x'], agent_position['z']), 
                   radius=0.2, color='red', fill=True)
    ax.add_patch(circle)
    
    # 添加方向指示（根据机器人的旋转角度）
    arrow_length = 0.5
    rotation_rad = np.deg2rad(agent_rotation['y'])
    dx = arrow_length * np.sin(rotation_rad)
    dz = arrow_length * np.cos(rotation_rad)
    ax.arrow(agent_position['x'], agent_position['z'], 
             dx, dz, head_width=0.2, head_length=0.2, fc='blue', ec='blue')
    
    # 标记环境中的物体
    try:
        object_count = 0
        for obj in event.metadata['objects']:
            if 'position' in obj:
                obj_position = obj['position']
                # 使用不同颜色区分不同类型的物体
                color = 'gray'
                if 'objectType' in obj:
                    obj_type = obj['objectType']
                    if any(furniture in obj_type.lower() for furniture in ['sofa', 'table', 'bed', 'chair', 'cabinet']):
                        color = 'brown'
                    elif any(appliance in obj_type.lower() for appliance in ['fridge', 'tv', 'television', 'microwave']):
                        color = 'blue'
                    elif any(container in obj_type.lower() for container in ['drawer', 'cabinet', 'shelf']):
                        color = 'green'
                
                # 绘制物体位置
                circle = Circle((obj_position['x'], obj_position['z']), 
                              radius=0.15, color=color, fill=True, alpha=0.5)
                ax.add_patch(circle)
                
                # 添加物体标签（如果有）
                if 'objectType' in obj:
                    ax.text(obj_position['x'], obj_position['z'], obj['objectType'], 
                           fontsize=8, ha='center', va='center')
                
                object_count += 1
    except Exception as e:
        st.error(f"标记物体时出错: {str(e)}")
    
    # 将图形转换为图像
    fig.canvas.draw()
    try:
        # 尝试使用旧版本的方法
        map_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        map_img = map_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    except AttributeError:
        # 如果失败，尝试使用新版本的方法
        w, h = fig.canvas.get_width_height()
        # 尝试buffer_rgba
        try:
            buf = fig.canvas.buffer_rgba()
            map_img = np.asarray(buf, dtype=np.uint8)
            map_img = map_img.reshape(h, w, 4)
            # 转换为RGB（删除Alpha通道）
            map_img = map_img[:, :, :3]
        except AttributeError:
            # 如果buffer_rgba也不存在，则使用其他备选方法
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            buf = canvas.renderer.buffer_rgba()
            map_img = np.asarray(buf, dtype=np.uint8)
            map_img = map_img.reshape(h, w, 4)
            # 转换为RGB（删除Alpha通道）
            map_img = map_img[:, :, :3]
    
    plt.close(fig)
    
    return map_img

# 初始化会话状态
if 'env' not in st.session_state:
    # 创建 playground/views 目录
    if not os.path.exists('playground/views'):
        os.makedirs('playground/views')
    else:
        # 清空目录
        shutil.rmtree('playground/views')
        os.makedirs('playground/views')
    
    # 初始化环境
    floor_id = 'FloorPlan_Train1_5'#'FloorPlan205'  # 默认场景
    st.session_state.env = ThorEnvDogView(floor_id)
    
    # 初始化图像和动作历史
    initial_event = st.session_state.env.get_last_event()
    initial_image = initial_event.cv2img
    cv2.imwrite('playground/views/view_init.png', initial_image)
    
    st.session_state.view_memory = [initial_image]
    st.session_state.act_memory = ["Init"]
    st.session_state.step_counter = 0
    st.session_state.action = None

# 页面标题
st.title("AI2Thor 环境交互控制器")

# 创建两列布局
col1, col2 = st.columns([1, 4])

with col1:
    # 动作选择
    st.write("选择动作：")
    col_actions1, col_actions2, col_actions3 = st.columns(3)
    
    with col_actions1:
        if st.button("MoveAhead"):
            st.session_state.action = "MoveAhead"
        if st.button("MoveLeft"):
            st.session_state.action = "MoveLeft"
    
    with col_actions2:
        if st.button("RotateRight"):
            st.session_state.action = "RotateRight"
        if st.button("RotateLeft"):
            st.session_state.action = "RotateLeft"
    
    with col_actions3:
        if st.button("MoveRight"):
            st.session_state.action = "MoveRight"
        if st.button("Done"):
            st.session_state.action = "Done"
    
    # 显示当前选择的动作
    st.write(f"当前选择的动作: {st.session_state.action if 'action' in st.session_state else '无'}")

    degrees = st.number_input("输入旋转角度（度）", min_value=1.0, max_value=180.0, value=30.0, step=1.0)
    magnitude = st.number_input("输入移动幅度", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
    
    # 执行按钮
    execute = st.button("执行动作")
    
    # 显示历史动作
    st.subheader("动作历史")
    st.write(st.session_state.act_memory)
    
    # 重置环境
    if st.button("重置环境"):
        floor_id = st.text_input("场景ID", "FloorPlan205")
        st.session_state.env.reset(floor_id)
        
        # 清空历史
        st.session_state.view_memory = [st.session_state.env.get_last_event().cv2img]
        st.session_state.act_memory = ["Init"]
        st.session_state.step_counter = 0
        st.session_state.action = None
        
        # 保存初始图像
        initial_image = st.session_state.env.get_last_event().cv2img
        cv2.imwrite('playground/views/view_init.png', initial_image)
        st.experimental_rerun()

# 执行动作
if execute and 'action' in st.session_state and st.session_state.action:
    action = st.session_state.action
    # 执行相应的动作
    if 'Rotate' in action:
        event = st.session_state.env.step(action, degrees=degrees)
    elif 'Move' in action:
        event = st.session_state.env.step(action, magnitude=magnitude)
    else:
        event = st.session_state.env.step(action)
    
    # 更新计数器
    st.session_state.step_counter += 1
    
    # 获取新图像并保存
    new_image = event.cv2img
    image_path = f'playground/views/view_{st.session_state.step_counter}_{action}.png'
    cv2.imwrite(image_path, new_image)
    
    # 更新历史
    st.session_state.view_memory.append(new_image)
    st.session_state.act_memory.append(action)

# 在右侧列显示当前图像
with col2:
    # 创建两个子列用于并排显示第一人称视图和平面图
    view_col, map_col = st.columns(2)
    
    with view_col:
        st.subheader("第一人称视图")
        
        # 获取最新图像并显示
        latest_image = st.session_state.view_memory[-1]
        
        # OpenCV 图像是 BGR 格式，而 Streamlit 需要 RGB
        latest_image_rgb = cv2.cvtColor(latest_image, cv2.COLOR_BGR2RGB)
        st.image(latest_image_rgb, caption=f"动作后视图: {st.session_state.act_memory[-1]}")
        
    with map_col:
        st.subheader("平面图与机器人位置")
        
        # 获取当前事件和平面图
        event = st.session_state.env.get_last_event()
        try:
            map_image = draw_map_with_agent(event)
            st.image(map_image, caption="机器人在环境中的位置")
        except Exception as e:
            st.error(f"无法生成平面图: {str(e)}")
    
    # 显示环境状态信息
    st.subheader("环境信息")
    st.write(f"机器人位置: {event.metadata['agent']['position']}")
    st.write(f"机器人旋转: {event.metadata['agent']['rotation']}")
    
    # 显示可见对象
    visible_objects = [obj['objectId'] for obj in event.metadata['objects'] if obj['visible']]
    st.write("可见对象:")
    st.write(visible_objects)