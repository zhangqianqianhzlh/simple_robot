import numpy as np
from typing import List, Dict, Any, Optional
import json
import matplotlib.pyplot as plt
import cv2

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
        self.location_history: List[Dict[str, Any]] = []


        
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

    def add_location(self, location: List[float], step_number: int) -> None:
        """
        添加一个新位置到位置历史记录中
        
        Args:
            location: 包含 [x, y, z] 坐标的列表
        """
        self.location_history.append({
            "location": location,
            "step_number": step_number
        })

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

    def draw_map(self) -> np.ndarray:
        """
        根据action_history绘制机器人运动轨迹图，返回numpy数组格式的图像
        
        Returns:
            np.ndarray: BGR格式的图像数组，可直接用于cv2.imwrite
        """
        # 初始化起始位置
        current_pos = np.array([0, 0, 0])
        positions = [current_pos.copy()]  # 用于存储所有位置
        
        print(self.action_history)
        print(self.location_history)
        print("*"*50)
        # 从action_history中提取位置信息
        for action in self.action_history:
            print(action)
            if 'location' in action:
                pos = np.array(action['location'])
                positions.append(pos)
        
        # 转换位置列表为numpy数组，便于处理
        positions = np.array(positions)

        print(positions)
        
        # 创建图形
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        
        # 绘制轨迹
        ax.plot(positions[:, 0], positions[:, 1], 'b-', label='path', linewidth=2)
        
        # 绘制所有经过的点
        ax.scatter(positions[1:-1, 0], positions[1:-1, 1], c='yellow', marker='o', s=50, label='points')
        
        # 特别标记起点和终点
        ax.scatter(positions[0, 0], positions[0, 1], c='g', marker='o', s=100, label='starting point')
        ax.scatter(positions[-1, 0], positions[-1, 1], c='r', marker='x', s=100, label='current position')
        
        # 添加点的编号
        for i, pos in enumerate(positions):
            ax.annotate(f'{i}', (pos[0], pos[1]), xytext=(5, 5), textcoords='offset points')
        
        # 计算坐标范围
        if len(positions) > 1:
            x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
            y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
        else:
            # 如果只有一个点，以该点为中心创建范围
            x_min, x_max = positions[0, 0] - 1, positions[0, 0] + 1
            y_min, y_max = positions[0, 1] - 1, positions[0, 1] + 1
        
        # 添加边距（坐标范围的10%）
        x_margin = max((x_max - x_min) * 0.1, 0.2)  # 确保至少有0.2的边距
        y_margin = max((y_max - y_min) * 0.1, 0.2)
        
        # 扩展坐标范围，确保至少是2*2的正方形
        x_range = max(x_max - x_min + 2 * x_margin, 2.0)
        y_range = max(y_max - y_min + 2 * y_margin, 2.0)
        
        # 确保是正方形：取x和y范围的较大值
        square_range = max(x_range, y_range)
        
        # 计算新的x和y范围的中心点
        x_center = (x_max + x_min) / 2
        y_center = (y_max + y_min) / 2
        
        # 根据中心点和正方形范围计算新的边界
        new_x_min = x_center - square_range / 2
        new_x_max = x_center + square_range / 2
        new_y_min = y_center - square_range / 2
        new_y_max = y_center + square_range / 2
        
        # 设置显示范围为正方形
        ax.set_xlim(new_x_min, new_x_max)
        ax.set_ylim(new_y_min, new_y_max)
        
        # 设置图形属性
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Robot Path')
        ax.grid(True)
        ax.legend()
        
        # 保持坐标轴比例相等
        ax.set_aspect('equal')
        
        # 将matplotlib图形转换为numpy数组
        fig.canvas.draw()
        
        # 获取图形的RGB数据 (使用新的API)
        img_data = np.asarray(fig.canvas.buffer_rgba())
        
        # 转换RGBA为RGB
        img_data = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)
        
        # 关闭matplotlib图形，释放内存
        plt.close(fig)
        
        return img_data
