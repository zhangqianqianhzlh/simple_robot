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

def main():
    st.set_page_config(page_title="Robot Navigation Simulation", layout="wide")
    st.title("VLM Navigation Simulation")

    # Sidebar for configuration
    st.sidebar.header("Simulation Settings")
    
    # Floor plan selection
    floor_id = st.sidebar.selectbox(
        "Select Floor Plan", 
        ["FloorPlan10", "FloorPlan201", "FloorPlan301", "FloorPlan401"],
        index=0
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
    if 'env' not in st.session_state:
        st.session_state.env = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'step_number' not in st.session_state:
        st.session_state.step_number = 0
    if 'view' not in st.session_state:
        st.session_state.view = None
    if 'action_memory' not in st.session_state:
        st.session_state.action_memory = []
    if 'completed' not in st.session_state:
        st.session_state.completed = False
    if 'augmented_images' not in st.session_state:
        st.session_state.augmented_images = []  # Store augmented images

    # Initialize vlm_output_str with a default value
    vlm_output_str = "No output available."

    # Main display area
    main_container = st.container()

    # Start simulation
    if start_button:
        with st.spinner("Initializing simulation..."):
            reset_simulation_state()
            st.session_state.model, st.session_state.env = initialize_simulation(floor_id)
            st.session_state.view = st.session_state.env.get_last_event().cv2img
            st.session_state.action_memory.append("Init")
            setup_views_directory()
            main_container.header("Simulation Started")
            main_container.image(cv2_to_streamlit_image(st.session_state.view), caption="Initial View", use_container_width=True)

    # Stop simulation
    if stop_button:
        st.session_state.running = False
        main_container.warning("Simulation stopped by user")

    # Run simulation steps
    if st.session_state.running and not st.session_state.completed:
        # Ensure environment and model are initialized
        if st.session_state.env is None or st.session_state.model is None:
            st.error("Environment or model not initialized")
            st.session_state.running = False
            return

        # Continue until max steps or stopped
        if st.session_state.step_number < max_steps:
            # Create a new container for each step
            step_container = main_container.container()
            step_container.subheader(f"Step {st.session_state.step_number + 1}")

            # Set up columns for augmented view and model's output
            col1, col2 = step_container.columns(2)

            # Get action proposals
            augmented_view, actions_info = get_action_proposals(st.session_state.view, api_url)

            if augmented_view is None or actions_info is None:
                step_container.error("Failed to get action proposals, skipping step")
                st.session_state.step_number += 1
                time.sleep(step_delay)
                st.rerun()

            # Display augmented view
            with col1:
                step_container.image(cv2_to_streamlit_image(augmented_view), caption="Augmented View with Action Paths", width=300)
                # Save the augmented view
                cv2.imwrite(f'views/view_{st.session_state.step_number}_augmented.png', augmented_view)
                st.session_state.augmented_images.append(augmented_view)  # Store the augmented image

            # Create string representation of available actions
            available_actions = [str(action["action_number"]) for action in actions_info]
            extended_available_actions = available_actions + ['0', 'Done']
            actions_str = ", ".join(extended_available_actions)

            # Format the prompt
            prompt = task_prompt.format(TARGET=target, ACTIONS=actions_str)

            with st.spinner("Getting VLM decision..."):
                # Get action from VLM using the augmented image
                vlm_output_str = st.session_state.model.run([augmented_view], model_id, prompt)
                reasoning, action_chosen = parse_vlm_output(vlm_output_str)

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
                        # Update the image in the container
                        step_container.image(cv2_to_streamlit_image(augmented_view), caption="Augmented View with Action Paths", width=300)
                except ValueError:
                    pass  # Skip if action_chosen is not a number

            # Display model's raw output and choice of action
            with col2:
                step_container.text("VLM Raw Output:")
                step_container.code(vlm_output_str, language=None)
                step_container.markdown("**VLM Reasoning (from JSON):**")
                step_container.info(reasoning)
                step_container.text(f"VLM Chose Action (from JSON): {action_chosen}")

            if action_chosen is None:
                st.error("VLM did not provide a valid action in the expected JSON format. Skipping step.")
                st.session_state.step_number += 1
                time.sleep(step_delay)
                st.rerun()

            elif isinstance(action_chosen, str) and action_chosen.lower() == 'done':
                st.success("Target reached, ending navigation (based on VLM 'Done' action).")
                st.session_state.completed = True
                st.session_state.running = False
            else:
                try:
                    action_number = int(action_chosen)
                    valid_action_numbers = [a["action_number"] for a in actions_info]
                    if action_number != 0 and action_number not in valid_action_numbers:
                         st.error(f"VLM chose an invalid action number: {action_number}. Available: {[0] + valid_action_numbers}. Skipping step.")
                         st.session_state.step_number += 1
                         time.sleep(step_delay)
                         st.rerun()
                    else:
                        st.write(f"Attempting to execute action: {action_number}")
                        event = execute_action(st.session_state.env, action_number, actions_info)

                        if event is None:
                            st.error("Failed to execute action (execute_action returned None), skipping step")
                            st.session_state.step_number += 1
                            time.sleep(step_delay)
                            st.rerun()
                        elif not event.metadata['lastActionSuccess']:
                            st.error(f"Action {action_number} failed in environment: {event.metadata['errorMessage']}. Skipping step.")
                            st.session_state.step_number += 1
                            time.sleep(step_delay)
                            st.rerun()
                        else:
                            new_view = event.cv2img

                            with col2:
                                st.image(cv2_to_streamlit_image(new_view), caption=f"After Action {action_number}", use_container_width=True)

                            cv2.imwrite(f'views/view_{st.session_state.step_number}_after_action_{action_number}.png', new_view)
                            st.session_state.action_memory.append(f"Action {action_number} (Reasoning: {reasoning})")

                            st.session_state.view = new_view
                            st.session_state.step_number += 1

                            time.sleep(step_delay)

                            if st.session_state.running:
                                st.rerun()

                except ValueError:
                    st.error(f"Invalid action '{action_chosen}' (expected integer string, 'Done', or '0'), skipping step")
                    st.session_state.step_number += 1
                    time.sleep(step_delay)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error processing or executing action {action_chosen}: {e}")
                    st.session_state.step_number += 1
                    time.sleep(step_delay)
                    st.rerun()
        else:
            st.session_state.running = False
            st.session_state.completed = True
            main_container.warning("Maximum steps reached.")
        
        # Display action history with images
        if st.session_state.step_number > 0:
            steps_per_page = 20
            total_steps = len(st.session_state.action_memory)
            total_pages = (total_steps + steps_per_page - 1) // steps_per_page

            page_number = st.sidebar.number_input("Page", min_value=1, max_value=total_pages, value=1)
            start_index = (page_number - 1) * steps_per_page
            end_index = start_index + steps_per_page

            with st.expander("Action History"):
                for i, (action_str, image) in enumerate(zip(st.session_state.action_memory[start_index:end_index], st.session_state.augmented_images[start_index:end_index]), start=start_index):
                    st.text(f"Step {i}: {action_str}")
                    st.image(cv2_to_streamlit_image(image), caption=f"Augmented View for Step {i}", width=300)

    # Example of folding VLM raw output
    with st.expander("VLM Raw Output"):
        st.code(vlm_output_str, language=None)

if __name__ == "__main__":
    main() 