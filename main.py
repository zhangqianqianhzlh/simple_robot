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
    task_prompt = "You are a robot that is navigating a house. You are given a list of last N image of your first person view from the oldest to the most recent. You current task is to find {OBJ} in the room and get in front of it, meaning the object should be in the center of the image and occupy most of the image. Given this task, you need to choose your next action from the following options: {ACTIONS}. Output only the action you choose, no other text. When you are in front of the object, you should stop and output 'Done'."
    target = "a white fridge on the left"
    buffer_len = 5
    memory = []

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

    actions = ['MoveAhead', 'RotateRight', 'RotateLeft', 'MoveLeft', 'MoveRight']
    event = controller.step(action="Done") # ge tthe initial image
    view = event.cv2img
    memory.append(view)

    if not os.path.exists('views'):
        os.makedirs('views')
    else:
        # remove the view folder and make a new one
        shutil.rmtree('views')
        os.makedirs('views')

    for i in range(50):
        # keep the last buffer_len views in memory
        memory = memory[-buffer_len:]

        a = model.run(memory, model_id, task_prompt.format(OBJ=target, ACTIONS=actions))
        print(a, len(memory))

        if 'Rotate' in a:
            event = controller.step(action=a, degrees=30)
        elif 'Done' in a:
            print("I am done")
            break
        else:
            event = controller.step(action=a)

        view = event.cv2img
        cv2.imwrite(f'views/view_{i}_{a}.png', view)

        memory.append(view)

