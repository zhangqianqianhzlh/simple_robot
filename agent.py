import cv2
import os
import shutil
import base64
import requests
import json
import numpy as np
import re
from models import VLM
from env import ThorEnvDogView

class VLMNavigationAgent:
    def __init__(self, env, model_id="Pro/Qwen/Qwen2.5-VL-7B-Instruct", api_url="http://10.8.25.28:8075/generate_action_proposals", max_distance_to_move=1.0):
        self.model = VLM("llm.yaml")
        self.env = env
        self.model_id = model_id
        self.api_url = api_url
        self.step_number = 0
        self.action_memory = []
        self.augmented_images = []
        self.view = None
        self.depth = None
        self.completed = False
        self.vlm_output_str = "No output available."
        self.max_distance_to_move = max_distance_to_move
        self.last_actions_info = None
        self.depth_memory = []
        self.setup_views_directory()

    def setup_views_directory(self):
        """Ensure the views directory is ready for storing images."""
        if not os.path.exists('views'):
            os.makedirs('views')
        else:
            shutil.rmtree('views')
            os.makedirs('views')

    def convert_image_to_base64(self, image):
        """Convert an image to base64 format."""
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')

    def send_api_request(self, image_base64):
        """Send a request to the action proposal API."""
        payload = {
            "image": image_base64,
            "min_angle": 20,
            "number_size": 30,
            "min_path_length": 150,
            "min_arrow_width": 30
        }
        response = requests.post(self.api_url, json=payload)
        return response

    def handle_api_response(self, response):
        """Handle the API response and return the augmented image and actions."""
        if response.status_code == 200:
            data = response.json()
            augmented_image_base64 = data["image"]
            augmented_image_bytes = base64.b64decode(augmented_image_base64)
            augmented_image_np = cv2.imdecode(np.frombuffer(augmented_image_bytes, np.uint8), cv2.IMREAD_COLOR)
            actions = data["actions"]
            return augmented_image_np, actions
        else:
            raise Exception(f"Error: API returned status code {response.status_code}")

    def get_action_proposals(self, image):
        """Get action proposals from the API."""
        image_base64 = self.convert_image_to_base64(image)
        response = self.send_api_request(image_base64)
        return self.handle_api_response(response)

    def parse_vlm_output(self, vlm_output_str):
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
            raise Exception("Failed to parse VLM output as JSON.")
        return reasoning, action_chosen

    def execute_action(self, action_number, actions_info):
        """
        Execute the chosen action by rotating to the appropriate degree and moving forward
        """
        # Find the action info for the chosen action number
        action_info = next((a for a in actions_info if a["action_number"] == action_number), None)
        
        if action_info is None:
            raise Exception(f"Error: Action {action_number} not found in action proposals")
        
        degree = action_info["turning_degree"]
        
        # If action is 0, turn around 180 degrees
        if action_number == 0:
            event = self.env.step("RotateRight", degrees=180)
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
        event = self.env.step(rotation_action, degrees=abs(degree))
        
        # Calculate move distance based on boundary distance
        if action_info["boundary_point"] is not None and self.depth is not None:
            boundary_x, boundary_y = action_info["boundary_point"]
            boundary_distance = self.depth[boundary_y, boundary_x]  # Get depth in meters
            move_distance = min(2/3 * boundary_distance, self.max_distance_to_move)
            # Move forward with calculated distance
            event = self.env.step("MoveAhead", magnitude=move_distance)
        else:
            # If no boundary point or depth info, use default movement
            event = self.env.step("MoveAhead")
            
        return event

    def reset(self):
        """Reset the agent state for a new simulation."""
        self.step_number = 0
        self.action_memory = []
        self.augmented_images = []
        self.depth_memory = []
        self.completed = False
        event = self.env.get_last_event()
        self.view = event.cv2img
        self.depth = event.depth_frame
        self.depth_memory.append(self.depth)
        self.action_memory.append("Init")
        # Add a placeholder augmented image to align with action_memory
        self.augmented_images.append(self.view)  # Use original view as placeholder for initial state
        self.setup_views_directory()

    def step(self, target, task_prompt, max_steps=50):
        """
        Execute one step of the navigation process
        Returns: (augmented_view, actions_info, reasoning, action_chosen, new_view)
        """
        if self.completed or self.step_number >= max_steps:
            print(f"Step skipped: completed={self.completed}, step_number={self.step_number}, max_steps={max_steps}")
            return None, None, None, None, None

        try:
            print("Getting action proposals...")
            # Get action proposals
            augmented_view, actions_info = self.get_action_proposals(self.view)
            if augmented_view is None or actions_info is None:
                print("Failed to get action proposals")
                return None, None, None, None, None
                
            print(f"Got {len(actions_info)} action proposals")
            self.last_actions_info = actions_info  # Store the actions info for history display
            
            # Create string representation of available actions
            available_actions = [str(action["action_number"]) for action in actions_info]
            extended_available_actions = available_actions + ['0', 'Done']
            actions_str = ", ".join(extended_available_actions)

            # Format the prompt
            prompt = task_prompt.format(TARGET=target, ACTIONS=actions_str)
            print("Getting VLM output...")
            # Get action from VLM using the augmented image
            self.vlm_output_str = self.model.run([augmented_view], self.model_id, prompt)
            print(f"VLM output: {self.vlm_output_str}")
            reasoning, action_chosen = self.parse_vlm_output(self.vlm_output_str)

            if action_chosen is None:
                print("Failed to parse VLM output")
                return augmented_view, actions_info, "Failed to parse VLM output", None, None

            if isinstance(action_chosen, str) and action_chosen.lower() == 'done':
                print("Task completed")
                self.completed = True
                self.step_number += 1
                self.action_memory.append(f"Done (Reasoning: {reasoning})")
                self.augmented_images.append(augmented_view)
                self.depth_memory.append(self.depth)
                return augmented_view, actions_info, reasoning, action_chosen, self.view

            try:
                action_number = int(action_chosen)
                valid_action_numbers = [a["action_number"] for a in actions_info]
                if action_number != 0 and action_number not in valid_action_numbers:
                    print(f"Invalid action number: {action_number}")
                    return augmented_view, actions_info, f"Invalid action number: {action_number}", None, None

                print(f"Executing action {action_number}")
                event = self.execute_action(action_number, actions_info)
                if event is None:
                    print("Failed to execute action")
                    return augmented_view, actions_info, "Failed to execute action", None, None
                if not event.metadata['lastActionSuccess']:
                    print(f"Action failed: {event.metadata['errorMessage']}")
                    return augmented_view, actions_info, f"Action failed: {event.metadata['errorMessage']}", None, None

                new_view = event.cv2img
                self.depth = event.depth_frame
                self.depth_memory.append(self.depth)
                self.view = new_view
                self.step_number += 1
                self.action_memory.append(f"Action {action_number} (Reasoning: {reasoning})")
                self.augmented_images.append(augmented_view)  # Only update augmented_images when action is successful
                print(f"Step completed successfully. New step number: {self.step_number}")

                return augmented_view, actions_info, reasoning, action_chosen, new_view

            except ValueError as e:
                print(f"ValueError in action execution: {str(e)}")
                return augmented_view, actions_info, f"Invalid action format: {action_chosen}", None, None

        except Exception as e:
            print(f"Exception in step: {str(e)}")
            return None, None, f"Error during step: {str(e)}", None, None
