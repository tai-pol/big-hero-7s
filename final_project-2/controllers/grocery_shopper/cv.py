import numpy as np
import cv2
import math
from ultralytics import YOLO  
from controller import Robot, Camera, Keyboard

#TRAINED MODEL
model = YOLO('best.pt') 

CONF = .25 
CUBE_ID = 0 
  
def run_cv(camera, depth_cam):
    # configurations for camera matrix
    raw = camera.getImage()
    CAMERA_HEIGHT = camera.getHeight()
    CAMERA_WIDTH = camera.getWidth()
    bgra  = np.frombuffer(raw, np.uint8).reshape((CAMERA_HEIGHT, CAMERA_WIDTH, 4))
    bgr   = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
    depth_W = depth_cam.getWidth()
    depth_H = depth_cam.getHeight()

    result = model.predict(bgr, conf=CONF, verbose=False)[0]
    found = False

    for box in result.boxes:
        if int(box.cls[0]) != CUBE_ID:
            continue
        found = True
        # top left and bottom right of detected object
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # center of found cube
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(bgr, (cx, cy), 4, (255, 0, 255), -1)

        # normalize to get scalable center of cube 
        u = (cx + 0.5) / CAMERA_WIDTH
        v = (cy + 0.5) / CAMERA_HEIGHT 
        FIELD_OF_VIEW = camera.getFov()

        # scale horizontal and vert position in depth image
        cx_s = u * depth_W
        cy_s = v * depth_H
        # rotate depth image 90 degrees
        cx_d = int(depth_H - 1 - cy_s)
        cy_d = int(cx_s)
        # clamp values
        cx_d = max(0, min(depth_W - 1, cx_d))
        cy_d = max(0, min(depth_H - 1, cy_d))

        depth_img = depth_cam.getRangeImageArray()
        depth = depth_img[cx_d][cy_d]
        if math.isinf(depth) or depth <= 0.0:
            continue
        
        # normalized coords -1 - 1
        nx = (cx - CAMERA_WIDTH/2) / (CAMERA_WIDTH/2)
        ny = (cy - CAMERA_HEIGHT/2) / (CAMERA_HEIGHT/2)
        x_c = depth
        y_c = -nx * depth * np.tan(FIELD_OF_VIEW/2)
        z_c = -ny * depth * np.tan(FIELD_OF_VIEW/2)
        pt_cam = np.array([x_c, y_c, z_c, 1.0])


        cv2.imshow("Cube Detection", bgr)
        cv2.waitKey(1)
        return True

    return False, None

