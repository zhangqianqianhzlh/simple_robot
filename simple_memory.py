import numpy as np
from typing import List, Dict, Any, Optional
import json

class SimpleMemory:
    def __init__(self, max_memory_size: int = 10):
        """
        Initialize the memory module.
        
        Args:
            max_memory_size: Maximum number of recent actions to remember
        """
        self.max_memory_size = max_memory_size
        self.action_history: List[Dict[str, Any]] = []
        self.explored_areas = set()  # Track explored areas to avoid revisiting
        self.repeating_pattern_threshold = 3  # Number of repeated actions to consider as a pattern
        self.completion_checks: List[Dict[str, Any]] = []  # Store task completion checks
        
    def add_action(self, action_info: Dict[str, Any]) -> None:
        """
        Add a new action to the memory.
        
        Args:
            action_info: Dictionary containing action details including:
                        - action_number: The action taken
                        - reasoning: The reasoning behind the action
                        - step_number: The step number when action was taken
        """
        self.action_history.append(action_info)
        # Keep only the most recent actions
        if len(self.action_history) > self.max_memory_size:
            self.action_history.pop(0)
            
    def add_completion_check(self, check_info: Dict[str, Any]) -> None:
        """
        Add a task completion check to the memory.
        
        Args:
            check_info: Dictionary containing completion check details including:
                       - completed: Whether the task was completed
                       - reasoning: The reasoning behind the completion check
                       - step_number: The step number when check was performed
        """
        self.completion_checks.append(check_info)
        # Keep only the most recent checks
        if len(self.completion_checks) > self.max_memory_size:
            self.completion_checks.pop(0)
            
    def update_explored_area(self, view: np.ndarray, depth: np.ndarray) -> None:
        """
        Update the memory with information about the current area.
        
        Args:
            view: Current RGB view
            depth: Current depth information
        """
        # Create a simple hash of the view to track explored areas
        # This is a simplified version - in practice you might want to use more sophisticated
        # methods to determine if an area has been explored
        view_hash = hash(view.tobytes())
        self.explored_areas.add(view_hash)
        
    def is_repeating_pattern(self) -> bool:
        """
        Check if the agent is stuck in a repeating pattern of actions.
        
        Returns:
            bool: True if a repeating pattern is detected
        """
        if len(self.action_history) < self.repeating_pattern_threshold:
            return False
            
        recent_actions = [action["action_number"] for action in self.action_history[-self.repeating_pattern_threshold:]]
        return len(set(recent_actions)) == 1  # All actions are the same
        
    def get_memory_summary(self, last_n: int = 5) -> str:
        """
        Generate a summary of the memory for the VLM.
        
        Returns:
            str: A formatted string summarizing the memory
        """
        if not self.action_history and not self.completion_checks:
            return "No previous actions or completion checks recorded."
            
        summary = "Previous Actions:\n"
        for action in self.action_history[-last_n:]:
            summary += f"- Step {action['step_number']}: Action {action['action_number']} (Reasoning: {action['reasoning']})\n"
            
        if self.completion_checks:
            summary += "\nRecent Task Completion Checks:\n"
            for check in self.completion_checks[-last_n:]:
                status = "Completed" if check['completed'] else "Not Completed"
                summary += f"- Step {check['step_number']}: {status} (Reasoning: {check['reasoning']})\n"
            
        if self.is_repeating_pattern():
            summary += "\nWARNING: Agent appears to be stuck in a repeating pattern of actions.\n"
            
        return summary
        
    def get_recent_actions(self, last_n: int = 5) -> str:
        """
        Get a string representation of recent actions.
        
        Returns:
            str: Formatted string of recent actions
        """
        if not self.action_history:
            return "No recent actions."
            
        recent = self.action_history[-last_n:]  # Get last n actions
        return "\n".join([f"Step {a['step_number']}: Action {a['action_number']}" for a in recent])
        
    def reset(self) -> None:
        """Reset the memory to its initial state."""
        self.action_history = []
        self.explored_areas = set()
        self.completion_checks = []
