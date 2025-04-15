from ai2thor.controller import Controller
import numpy as np
import cv2
import os
import shutil
from models import VLM
from env import ThorEnvDogView



if __name__ == "__main__":
    model = VLM("llm.yaml")
    floor_id = 'FloorPlan_Train1_5'
    model_id = "Qwen/Qwen2.5-VL-72B-Instruct"
    task_prompt = "You are a robot that is navigating in a house. You are given a list of last N image of your first person views from the oldest to the most recent. Your previous action history is {ACTION_HISTORY} from the oldest to the most recent. You current task is to follow the instructions: '{TARGET}' and get very close to it. When you cannot immediately find the object, you should rotate around the room to find it. Given this task, you need to choose your next action from the following options: {ACTIONS}. Output only the action you choose, no other text. Only output'Done' if the target object is in the center of the image and occupy most of the image."
    # target = "turn left and move forward and then find the white fridge on your right hand side"    
    target = "laptop"
    buffer_len = 5
    view_memory = []
    act_memory = []

    env = ThorEnvDogView(floor_id)

    if not os.path.exists('playground/views'):
        os.makedirs('playground/views')
    else:
        # remove the view folder and make a new one
        shutil.rmtree('playground/views')
        os.makedirs('playground/views')

    actions = ['MoveAhead', 'RotateRight', 'RotateLeft', 'MoveLeft', 'MoveRight', "Done"]
    view = env.get_last_event().cv2img # ge tthe initial image
    cv2.imwrite(f'playground/views/view_init.png', view)
    view_memory.append(view)
    act_memory.append("Init")

    event = env.get_last_event()
    print(f"机器人位置: {event.metadata['agent']['position']}")
    print(f"机器人旋转: {event.metadata['agent']['rotation']}")

    actions = ["RotateLeft", "MoveAhead", "Done"]
    for i in range(len(actions)):
        a = actions[i]
        if a == "MoveAhead":
            event = env.step("MoveAhead", magnitude=0.1)
        elif "Rotate" in a:
            event = env.step(a, degrees=90)
        else:
            event = env.step(a)
        
        event = env.get_last_event()
        view = event.cv2img
        print(f"finished {i} actions {actions[i]}")
        print(f"机器人位置: {event.metadata['agent']['position']}")
        print(f"机器人旋转: {event.metadata['agent']['rotation']}")
        cv2.imwrite(f'playground/views/view_{i}_{actions[i]}.png', view)
        view_memory.append(view)
        act_memory.append(actions[i])
