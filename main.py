import yaml
from models import VLM
from ai2thor.controller import Controller
import numpy as np
import cv2
import os
import shutil


if __name__ == "__main__":
    model = VLM("llm.yaml")
    floor_id = 'FloorPlan10'
    model_id = "Qwen/Qwen2.5-VL-32B-Instruct"
    task_prompt = "You are a robot that is navigating in a house. You are given a list of last N image of your first person views from the oldest to the most recent. Your previous action history is {ACTION_HISTORY} from the oldest to the most recent. You current task is to find {OBJ} and get very close to it. When you cannot immediately find the object, you should rotate around the room to find it. Given this task, you need to choose your next action from the following options: {ACTIONS}. Output only the action you choose, no other text. Only output'Done' if the target object is in the center of the image and occupy most of the image."
    target = "a white fridge on the left side of the room"
    buffer_len = 5
    view_memory = []
    act_memory = []

    controller = Controller(
        agentMode="default",
        visibilityDistance=1.5,
        scene=floor_id,

        # step sizes
        gridSize=0.25,
        snapToGrid=True,
        rotateStepDegrees=90,

        # image modalities
        renderDepthImage=False,
        renderInstanceSegmentation=False,

        # camera properties
        width=300,
        height=300,
        fieldOfView=90
    )

    actions = ['MoveAhead', 'RotateRight', 'RotateLeft', 'MoveLeft', 'MoveRight', "Done"]
    event = controller.step(action="Done") # ge tthe initial image
    view = event.cv2img
    view_memory.append(view)
    act_memory.append("Init")

    if not os.path.exists('views'):
        os.makedirs('views')
    else:
        # remove the view folder and make a new one
        shutil.rmtree('views')
        os.makedirs('views')

    for i in range(50):
        # keep the last buffer_len views in view_memory
        view_memory = view_memory[-buffer_len:]

        prompt = task_prompt.format(OBJ=target, ACTIONS=actions, ACTION_HISTORY=act_memory)
        a = model.run(view_memory, model_id, prompt)
        print(a, len(view_memory), len(act_memory), prompt)

        if 'Rotate' in a:
            event = controller.step(action=a, degrees=30)
        elif 'Done' in a:
            print("I am done")
            break
        else:
            event = controller.step(action=a)

        view = event.cv2img
        cv2.imwrite(f'views/view_{i}_{a}.png', view)

        view_memory.append(view)
        act_memory.append(a)

