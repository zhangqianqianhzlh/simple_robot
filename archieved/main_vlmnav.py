from models import VLM
import cv2
import os
import shutil
import base64
import requests
import json
import numpy as np
from env import ThorEnvDogView

def get_action_proposals(image, api_url="http://10.8.25.28:8075/generate_action_proposals"):
    """
    Get action proposals from the API
    """
    # Convert image to base64
    _, buffer = cv2.imencode('.jpg', image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Prepare request payload
    payload = {
        "image": image_base64,
        "min_angle": 20,
        "number_size": 30,
        "min_path_length": 100
    }
    
    # Send request to API
    response = requests.post(api_url, json=payload)
    
    if response.status_code == 200:
        data = response.json()
        # Get augmented image
        augmented_image_base64 = data["image"]
        augmented_image_bytes = base64.b64decode(augmented_image_base64)
        augmented_image_np = cv2.imdecode(np.frombuffer(augmented_image_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        # Get action proposals
        actions = data["actions"]
        return augmented_image_np, actions
    else:
        print(f"Error: API returned status code {response.status_code}")
        return None, None

def execute_action(env, action_number, actions_info):
    """
    Execute the chosen action by rotating to the appropriate degree and moving forward
    """
    # Find the action info for the chosen action number
    action_info = next((a for a in actions_info if a["action_number"] == action_number), None)
    
    if action_info is None:
        print(f"Error: Action {action_number} not found in action proposals")
        return None
    
    degree = action_info["turning_degree"]
    print(f"Executing action {action_number} with degree {degree}")
    
    # If action is 0, turn around 180 degrees
    if action_number == 0:
        # # Turn around (180 degrees) - do this in smaller steps
        # for _ in range(6):  # 6 * 30 = 180 degrees
        #     event = env.step("RotateRight")
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


if __name__ == "__main__":
    model = VLM("llm.yaml")
    floor_id = 'FloorPlan10'
    model_id = "Qwen/Qwen2.5-VL-32B-Instruct"
    api_url = "http://10.8.25.28:8075/generate_action_proposals"
    task_prompt = "You are a robot that is navigating in a house. You are looking at an image with numbered paths. Each number represents a possible direction you can take. Your current task is to follow the instructions: '{TARGET}' and get very close to it. Choose the number of the path that best helps you achieve your task. Output only the number you choose, no other text. Output 'Done' if the target object is in the center of the image and occupies most of the image. Output '0' if you need to turn around because you don't see a good path forward."
    target = "turn left and move forward and then find the white fridge on your right hand side"
    act_memory = []

    env = ThorEnvDogView(floor_id)

    view = env.get_last_event().cv2img
    act_memory.append("Init")

    if not os.path.exists('views'):
        os.makedirs('views')
    else:
        # remove the view folder and make a new one
        shutil.rmtree('views')
        os.makedirs('views')

    for i in range(50):
        print(f"================== Step {i} ==================")

        # Get action proposals for the current view
        augmented_view, actions_info = get_action_proposals(view, api_url)
        
        print(f"Actions: {actions_info}")

        if augmented_view is None or actions_info is None:
            print("Failed to get action proposals, skipping step")
            continue
        
        # Save the augmented view
        cv2.imwrite(f'views/view_{i}_augmented.png', augmented_view)
        
        # Create string representation of available actions
        available_actions = [str(action["action_number"]) for action in actions_info]
        available_actions.append("Done")
        actions_str = ", ".join(available_actions)
        
        # Format the prompt
        prompt = task_prompt.format(TARGET=target, ACTIONS=actions_str)
        
        # Get action from VLM using the augmented image
        a = model.run([augmented_view], model_id, prompt)
        print(f"Step {i}: VLM chose action {a}")
        
        if a == 'Done':
            print("Target reached, ending navigation")
            break
        
        try:
            # Convert action to integer
            action_number = int(a)
            
            # Execute the chosen action
            event = execute_action(env, action_number, actions_info)
            
            if event is None:
                print("Failed to execute action, skipping step")
                continue
            
            # Update view and action memory
            view = event.cv2img
            cv2.imwrite(f'views/view_{i}_after_action_{action_number}.png', view)
            act_memory.append(f"Action {action_number}")
            
        except ValueError:
            print(f"Invalid action '{a}', skipping step")
            continue

    print("FINISHED")

