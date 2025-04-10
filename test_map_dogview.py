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
        renderDepthImage=True,
        renderInstanceSegmentation=False,

        # camera properties
        width=720,
        height=480,
        fieldOfView=100,
    )

    actions = ['MoveAhead', 'RotateRight', 'RotateLeft', 'MoveLeft', 'MoveRight']
    
    if not os.path.exists('views_tmp'):
        os.makedirs('views_tmp')
    else:
        # remove the view folder and make a new one
        shutil.rmtree('views_tmp')
        os.makedirs('views_tmp')

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
        depth = event.depth_frame
        # depth is a 2D array of shape (480, 720), whose element is the distance to the nearest object in meters

        # Normalize depth values to 0-255 range and convert to uint8
        depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_uint8 = depth_normalized.astype(np.uint8)
        
        depth_heatmap = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)

        # save the view to a folder, if the folder doesn't exist, create it

        cv2.imwrite(f'views_tmp/view_{i}_{a}.png', view)
        cv2.imwrite(f'views_tmp/depth_{i}_{a}.png', depth_heatmap)

        break

    

        
        
