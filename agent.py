import cv2
import os
import shutil
import base64
import requests
import json
import numpy as np
import re
import json_repair
from models import VLM
from env import ThorEnvDogView
# from memory import NavigationMemory
from simple_memory import SimpleMemory

class VLMNavigationAgent:
    def __init__(self, env, model_id="Pro/Qwen/Qwen2.5-VL-7B-Instruct", api_url="http://10.8.25.28:8075/generate_action_proposals", max_distance_to_move=1.0):
        self.model = VLM("llm.yaml")
        self.env = env
        self.model_id = model_id
        self.api_url = api_url
        self.step_number = 0
        self.action_memory = []
        self.augmented_images = []
        self.complete_images = []  # Store complete (non-augmented) views
        self.view = None
        self.depth = None
        self.completed = False
        self.vlm_output_str = "No output available."
        self.max_distance_to_move = max_distance_to_move
        self.last_actions_info = None
        self.depth_memory = []
        self.memory = SimpleMemory()
        self.last_n_for_summary = 5
        self.last_n_for_done = 3  # Number of recent actions/images to consider for task completion check
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
            "min_path_length": 80,
            "min_arrow_width": 60
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
                    vlm_output_json = json_repair.loads(json_str)
                    reasoning = vlm_output_json.get("reasoning", "No reasoning provided in JSON.")
                    action_chosen = vlm_output_json.get("action", None)
            else:
                vlm_output_json = json_repair.loads(vlm_output_str.strip())
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
        self.complete_images = []  # Reset complete images
        self.depth_memory = []
        self.completed = False
        self.memory = SimpleMemory()  # Reset memory
        event = self.env.get_last_event()
        self.view = event.cv2img
        self.depth = event.depth_frame
        self.depth_memory.append(self.depth)
        self.action_memory.append("Initial View")
        self.augmented_images.append(self.view)
        self.complete_images.append(self.view)  # Store initial complete view
        self.setup_views_directory()

    def update_memory_with_action(self, action_number, actions_info, reasoning):
        """Update memory with the current action and state"""
        
        # Update explored area
        self.memory.update_explored_area(self.view, self.depth)
        
        # Record the action
        action_record = {
            "action_number": action_number,
            "reasoning": reasoning,
            "step_number": self.step_number
        }
        self.memory.add_action(action_record)
        
        # Check for repeating patterns
        if self.memory.is_repeating_pattern():
            print("Warning: Agent detected in repeating pattern of actions")

    def check_task_completion(self, target, current_view):
        """Check if the task is completed using a separate VLM call"""
        # Get recent history for context
        recent_actions = self.action_memory[-self.last_n_for_done:] if self.action_memory else []
        recent_images = self.augmented_images[-self.last_n_for_done:] if self.augmented_images else []
        
        # Create a detailed history of actions and their corresponding views
        history_context = []
        for i, (action, image) in enumerate(zip(recent_actions, recent_images)):
            step_number = self.step_number - len(recent_actions) + i + 1
            history_context.append(f"Step {step_number}: Action taken was '{action}'")
        
        # Format the completion check prompt
        completion_prompt = f"""
        You are a robot that has been following these navigation instructions: {target}
        
        Navigation History (from oldest to most recent):
        {chr(10).join(history_context)}
        
        You will be shown a sequence of {len(recent_images)} images followed by the current view:
        1. The first {len(recent_images)} images show your previous views in chronological order
        2. The last image is your current view
        
        Each previous view corresponds to the action taken at that step. Analyze the sequence of images to understand:
        - How your position and orientation changed over time
        - Whether you're following the instructions correctly
        - If you have completed all steps in the instructions
        
        Based on this analysis, determine if the task is completed.
        The task is considered completed if you have successfully followed all steps in the instructions.
        
        Output your response as a JSON object with two keys:
        1. "completed": boolean (true if task is completed, false otherwise)
        2. "reasoning": string (explanation of why the task is or isn't completed, including references to specific steps or views that support your conclusion)
        
        Example: {{"completed": true, "reasoning": "In the current view (Step {self.step_number}), I can see the target object clearly in front of me. I have completed all steps: went left, moved forward, and found the fridge on my right."}}
        Example: {{"completed": true, "reasoning": "Although I cannot see the target object in the current view, I can see it in the previous views. From previous views, I can see that I have followed the instructions correctly: went left, moved forward, and the fridge is now on my right."}}
        Example: {{"completed": false, "reasoning": "The target object is not visible in the current view or previous views. In Step {self.step_number-1}, I turned left, and in Step {self.step_number-2}, I moved forward, but I still need to find the fridge."}}
        """
        
        # Run VLM with current view and recent images
        images_to_analyze = recent_images + [current_view]
        completion_output = self.model.run(images_to_analyze, self.model_id, completion_prompt)
        
        try:
            # Parse the completion check output
            completion_json = json_repair.loads(completion_output)
            is_completed = completion_json.get("completed", False)
            reasoning = completion_json.get("reasoning", "No reasoning provided")
            
            # Store the completion check in memory
            self.memory.add_completion_check({
                "completed": is_completed,
                "reasoning": reasoning,
                "step_number": self.step_number
            })
            
            return is_completed, reasoning
        except:
            # Store failed check in memory
            self.memory.add_completion_check({
                "completed": False,
                "reasoning": "Failed to parse completion check output",
                "step_number": self.step_number
            })
            return False, "Failed to parse completion check output"

    def step(self, target, task_prompt, max_steps=50):
        """
        Execute one step of the navigation process with memory integration
        Returns: (augmented_view, actions_info, reasoning, action_chosen, new_view, is_completed)
        """
        if self.completed or self.step_number >= max_steps:
            print(f"Step skipped: completed={self.completed}, step_number={self.step_number}, max_steps={max_steps}")
            return None, None, None, None, None, self.completed

        try:
            # First check if task is completed
            is_completed, completion_reasoning = self.check_task_completion(target, self.view)
            if is_completed:
                print(f"Task completed: {completion_reasoning}")
                self.completed = True
                self.step_number += 1
                self.augmented_images.append(self.view)
                self.action_memory.append(f"Done (Reasoning: {completion_reasoning})")
                self.complete_images.append(self.view)  # Store final complete view
                return None, None, completion_reasoning, "Done", self.view, True

            print("Getting action proposals...")
            augmented_view, actions_info = self.get_action_proposals(self.view)
            if augmented_view is None or actions_info is None:
                print("Failed to get action proposals")
                return None, None, None, None, None, False
                
            print(f"Got {len(actions_info)} action proposals")
            self.last_actions_info = actions_info
            
            # Create string representation of available actions
            available_actions = [str(action["action_number"]) for action in actions_info]
            extended_available_actions = available_actions + ['0']  # Add '0' for turning around
            actions_str = ", ".join(extended_available_actions)

            # Get memory summary for context
            memory_summary = self.memory.get_memory_summary(last_n=self.last_n_for_summary)

            # replace "{TARGET}" and "{ACTIONS}" in the task prompt with the target and actions_str
            task_prompt = task_prompt.replace("{TARGET}", target).replace("{ACTIONS}", actions_str)
            
            # Format the prompt with memory context and target
            enhanced_prompt = f"""
            {task_prompt}
            
            Memory Context:
            {memory_summary}
            
            Based on this context and the goal of reaching {target}, choose the most appropriate action from: {actions_str}
            Consider avoiding repeating patterns and previously visited areas.
            """
            
            print("Getting VLM output with memory context...")
            self.vlm_output_str = self.model.run([augmented_view], self.model_id, enhanced_prompt)
            print(f"VLM output: {self.vlm_output_str}")
            reasoning, action_chosen = self.parse_vlm_output(self.vlm_output_str)

            if action_chosen is None:
                print("Failed to parse VLM output")
                return augmented_view, actions_info, "Failed to parse VLM output", None, None, False

            try:
                action_number = int(action_chosen)
                valid_action_numbers = [a["action_number"] for a in actions_info]
                
                # Check if we're in a repeating pattern and the chosen action is part of it
                if self.memory.is_repeating_pattern():
                    recent_actions = [action["action_number"] for action in self.memory.action_history[-self.memory.repeating_pattern_threshold:]]
                    if action_number in recent_actions:
                        print(f"Warning: Chosen action {action_number} is part of repeating pattern. Selecting random alternative.")
                        # Get all valid actions except the repeating ones
                        alternative_actions = [a for a in valid_action_numbers if a not in recent_actions]
                        if alternative_actions:  # If there are alternatives
                            action_number = np.random.choice(alternative_actions)
                            reasoning = f"Randomly selected alternative action {action_number} to break repeating pattern"
                        else:  # If no alternatives, just use the original action
                            print("No alternative actions available, proceeding with original choice")
                
                if action_number != 0 and action_number not in valid_action_numbers:
                    print(f"Invalid action number: {action_number}")
                    return augmented_view, actions_info, f"Invalid action number: {action_number}", None, None, False

                print(f"Executing action {action_number}")
                event = self.execute_action(action_number, actions_info)
                if event is None:
                    print("Failed to execute action")
                    return augmented_view, actions_info, "Failed to execute action", None, None, False
                if not event.metadata['lastActionSuccess']:
                    print(f"Action failed: {event.metadata['errorMessage']}")
                    return augmented_view, actions_info, f"Action failed: {event.metadata['errorMessage']}", None, None, False

                self.action_memory.append(f"Action {action_number} (Reasoning: {reasoning})")
                self.augmented_images.append(augmented_view)
                self.complete_images.append(self.view)  # Store the complete view

                new_view = event.cv2img
                self.depth = event.depth_frame
                self.depth_memory.append(self.depth)
                self.view = new_view
                self.step_number += 1
                
                # Update memory with the current action
                self.update_memory_with_action(action_number, actions_info, reasoning)
                
                print(f"Step completed successfully. New step number: {self.step_number}")

                return augmented_view, actions_info, reasoning, action_chosen, new_view, False

            except ValueError as e:
                print(f"ValueError in action execution: {str(e)}")
                return augmented_view, actions_info, f"Invalid action format: {action_chosen}", None, None, False

        except Exception as e:
            print(f"Exception in step: {str(e)}")
            return None, None, f"Error during step: {str(e)}", None, None, False
