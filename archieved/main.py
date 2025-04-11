from models import VLM
import cv2
import os
import shutil
from env import ThorEnv


if __name__ == "__main__":
    model = VLM("llm.yaml")
    floor_id = 'FloorPlan10'
    model_id = "Qwen/Qwen2.5-VL-32B-Instruct"
    task_prompt = "You are a robot that is navigating in a house. You are given a list of last N image of your first person views from the oldest to the most recent. Your previous action history is {ACTION_HISTORY} from the oldest to the most recent. You current task is to follow the instructions: '{TARGET}' and get very close to it. When you cannot immediately find the object, you should rotate around the room to find it. Given this task, you need to choose your next action from the following options: {ACTIONS}. Output only the action you choose, no other text. Only output'Done' if the target object is in the center of the image and occupy most of the image."
    target = "turn left and move forward and then find the white fridge on your right hand side"
    buffer_len = 5
    view_memory = []
    act_memory = []

    env = ThorEnv(floor_id)

    actions = ['MoveAhead', 'RotateRight', 'RotateLeft', 'MoveLeft', 'MoveRight', "Done"]
    view = env.get_last_event().cv2img # ge tthe initial image
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

        prompt = task_prompt.format(TARGET=target, ACTIONS=actions, ACTION_HISTORY=act_memory)
        a = model.run(view_memory, model_id, prompt)
        print(a, len(view_memory), len(act_memory), prompt)
        if a == 'Done':
            break

        event = env.step(a)
        
        view = event.cv2img
        cv2.imwrite(f'views/view_{i}_{a}.png', view)

        view_memory.append(view)
        act_memory.append(a)


    print("FINISHED")

