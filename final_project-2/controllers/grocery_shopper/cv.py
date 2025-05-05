import numpy as np
import cv2
import math
from ultralytics import YOLO  
from controller import Robot, Camera, Keyboard



#initialising
robot = Robot()
timestep = int(robot.getBasicTimeStep())

camera = robot.getDevice('camera')
camera.enable(timestep)
camera.recognitionEnable(timestep) 

depth_cam = robot.getDevice('range-finder')
depth_cam.enable(timestep)
depth_W = depth_cam.getWidth()
depth_H = depth_cam.getHeight()

kbd = robot.getKeyboard()
kbd.enable(timestep)

robot_parts=[]
yaw_motor   = robot.getDevice("head_1_joint")
pitch_motor = robot.getDevice("head_2_joint")

yaw_sensor   = yaw_motor.getPositionSensor();   yaw_sensor.enable(timestep)
pitch_sensor = pitch_motor.getPositionSensor(); pitch_sensor.enable(timestep)
N_PARTS = 12

#idk where pan joint actaully is lol
T_base_pan = np.array([[1,0,0, 0.00],
                       [0,1,0, 0.00],
                       [0,0,1, 1.13],
                       [0,0,0, 1   ]])

T_tilt_cam = np.array([[1,0,0, 0.00],
                       [0,1,0, 0.00],
                       [0,0,1, 0.00],
                       [0,0,0, 1   ]])


part_names = ("head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
              "arm_2_joint",  "arm_3_joint",  "arm_4_joint",      "arm_5_joint",
              "arm_6_joint",  "arm_7_joint",  "wheel_left_joint", "wheel_right_joint")

target_pos = (-0.4, 0.0, 0.09, 0.07, 1.02, -3.16, 1.27, 1.32, 0.0, 1.41, 'inf', 'inf')

for i in range(N_PARTS-2):
    robot_parts.append(robot.getDevice(part_names[i]))
    robot_parts[i].setPosition(float(target_pos[i]))
    robot_parts[i].setVelocity(robot_parts[i].getMaxVelocity() / 2.0)

wheel_left  = robot.getDevice("wheel_left_joint")
wheel_right = robot.getDevice("wheel_right_joint")
wheel_left.setVelocity(0.0)           
wheel_right.setVelocity(0.0)
# set camera properties
FIELD_OF_VIEW = camera.getFov()
CAMERA_HEIGHT = camera.getHeight()
CAMERA_WIDTH = camera.getWidth()
FOCAL_LENGTH = camera.getFocalLength()



focal_length_px = (CAMERA_WIDTH / 2) / np.tan(FIELD_OF_VIEW / 2)

#TRAINED MODELLLLLLLL AYAYAYA
model = YOLO('best.pt') 

CONF = .25 
CUBE_ID = 0 

# camera matrix
K = np.array([
    [focal_length_px, 0, CAMERA_WIDTH / 2],
    [0, focal_length_px, CAMERA_HEIGHT / 2],
    [0, 0, 1]
])

    
def run_cv(camera, depth_cam):
    raw = camera.getImage()
    bgra  = np.frombuffer(raw, np.uint8).reshape((CAMERA_HEIGHT, CAMERA_WIDTH, 4))
    bgr   = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)

    result = model.predict(bgr, conf=CONF, verbose=False)[0]
    found = False

    for box in result.boxes:
        if int(box.cls[0]) != CUBE_ID:
            continue
        found = True
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(bgr, (cx, cy), 4, (255, 0, 255), -1)

        u = (cx + 0.5) / CAMERA_WIDTH
        v = (cy + 0.5) / CAMERA_HEIGHT 
        cx_s = u * depth_W
        cy_s = v * depth_H
        cx_d = int(depth_H - 1 - cy_s)
        cy_d = int(cx_s)
        cx_d = max(0, min(depth_W - 1, cx_d))
        cy_d = max(0, min(depth_H - 1, cy_d))

        depth_img = depth_cam.getRangeImageArray()
        depth = depth_img[cx_d][cy_d]
        if math.isinf(depth) or depth <= 0.0:
            continue

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


# print("Press **F** to scan the current frame for cubes…")
# while robot.step(timestep) != -1:
#     yaw   = yaw_sensor.getValue()     # radians
#     pitch = pitch_sensor.getValue() 

#     key = kbd.getKey()

#     if key not in (ord('F'), ord('f')):
#         continue


#     raw   = camera.getImage()
#     bgra  = np.frombuffer(raw, np.uint8).reshape((CAMERA_HEIGHT, CAMERA_WIDTH, 4))
#     bgr   = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)

#     result   = model.predict(bgr, conf=CONF, verbose=False)[0]
#     found    = False

#     for box in result.boxes:
#         if int(box.cls[0]) != CUBE_ID:
#             continue
#         found = True
#         # topleft and bottomright
#         x1,y1,x2,y2 = map(int, box.xyxy[0]); 

#         # center of found cube
#         cx,cy = (x1+x2)//2, (y1+y2)//2

#         cv2.rectangle(bgr,(x1,y1),(x2,y2),(0,255,0),2)
#         cv2.circle(bgr,(cx,cy),4,(0,0,255),-1)
#         cv2.circle(bgr, (cx, cy), 4, (255, 0, 255), 2)

#         # normalize to get center of cube 
#         u = (cx + 0.5) / CAMERA_WIDTH
#         v = (cy + 0.5) / CAMERA_HEIGHT 

#         # scale horizontal and vert position in depth image
#         cx_s = u * depth_W
#         cy_s = v * depth_H

#         # rotate 90
#         cx_d = int(depth_H - 1 - cy_s) 
#         cy_d = int(cx_s) 

#         # clamp
#         cx_d = max(0, min(depth_W  - 1, cx_d))
#         cy_d = max(0, min(depth_H - 1, cy_d))

#         print(f"RGB centre ({cx},{cy})  ->  depth index ({cx_d},{cy_d})")

#         # normalized coords -1 - 1
#         nx = (cx - CAMERA_WIDTH/2)  / (CAMERA_WIDTH/2)
#         ny = (cy - CAMERA_HEIGHT/2) / (CAMERA_HEIGHT/2)

#         yaw   = yaw_sensor.getValue()
#         pitch = pitch_sensor.getValue()

#         Rz = np.array([[ np.cos(yaw), 0, np.sin(yaw), 0],
#                     [ 0,           1, 0,           0],
#                     [-np.sin(yaw), 0, np.cos(yaw), 0],
#                     [ 0,           0, 0,           1]])
#         Ry = np.array([[ 1, 0,            0,             0],
#                     [ 0, np.cos(pitch),-np.sin(pitch), 0],
#                     [ 0, np.sin(pitch), np.cos(pitch), 0],
#                     [ 0, 0,            0,             1]])
#         T_base_cam = T_base_pan @ Rz @ Ry @ T_tilt_cam


#         depth_img = depth_cam.getRangeImageArray()
#         valid_depths = [d for row in depth_img for d in row if not math.isinf(d) and d > 0.01]
#         print(f"Depth map: {len(valid_depths)} valid values, median = {np.median(valid_depths):.2f} m")
#         depth = depth_img[cy_d][cx_d]  


#         if math.isinf(depth) or depth <= 0.0:
#             continue

#         x_c = depth
#         y_c = -nx * depth * np.tan(FIELD_OF_VIEW/2)
#         z_c = -ny * depth * np.tan(FIELD_OF_VIEW/2)
#         pt_cam  = np.array([x_c, y_c, z_c, 1.0])

#         pt_base = T_base_cam @ pt_cam
#         print("cube in robot frame XYZ:", np.round(pt_base[:3],3))
#     if found:
#         cv2.imshow("Cube Detection", bgr)
        
#         cv2.waitKey(3)

#     else:
#         continue

#     pass
