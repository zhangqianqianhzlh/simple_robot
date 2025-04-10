from ai2thor.controller import Controller
import numpy as np
import cv2
import os
import shutil

if __name__ == "__main__":

    controller = Controller(
        agentMode="default",
        visibilityDistance=1.5,
        scene="FloorPlan10",

        # step sizes
        gridSize=0.25,
        snapToGrid=True,
        rotateStepDegrees=90,

        # image modalities
        renderDepthImage=False,
        renderInstanceSegmentation=False,

        # camera properties
        width=720,
        height=480,
        fieldOfView=100,
    )

    actions = ['MoveAhead', 'RotateRight', 'RotateLeft', 'MoveLeft', 'MoveRight']
    
    if not os.path.exists('views'):
        os.makedirs('views')
    else:
        # remove the view folder and make a new one
        shutil.rmtree('views')
        os.makedirs('views')

    # crouch to the ground, which is about 0.22 meters lower than the initial height
    controller.step(action="Crouch")
    controller.step("LookDown")

    for i in range(100):
        a = np.random.choice(actions)
        if 'Rotate' in a:
            event = controller.step(action=a, degrees=15)
        else:
            event = controller.step(action=a)

        view = event.cv2img
        # save the view to a folder, if the folder doesn't exist, create it

        cv2.imwrite(f'views/view_{i}_{a}.png', view)


    

        
        
