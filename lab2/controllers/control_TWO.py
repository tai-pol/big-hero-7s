"""csci3302_lab2 controller."""

# You may need to import some classes of the controller module.
import math
from controller import Robot, Motor, DistanceSensor
# import os

# Ground Sensor Measurements under this threshold are black
# measurements above this threshold can be considered white.
# TODO: Fill this in with a reasonable threshold that separates "line detected" from "no line detected"
GROUND_SENSOR_THRESHOLD = 0

# These are your pose values that you will update by solving the odometry equations
pose_x = 0
pose_y = 0
pose_theta = 0

# Index into ground_sensors and ground_sensor_readings for each of the 3 onboard sensors.
LEFT_IDX = 0
CENTER_IDX = 1
RIGHT_IDX = 2

# create the Robot instance.
robot = Robot()

# ePuck Constants
EPUCK_AXLE_DIAMETER = 0.053 # ePuck's wheels are 53mm apart.
EPUCK_MAX_WHEEL_SPEED = .11637 # TODO: To be filled in with ePuck wheel speed in m/s
MAX_SPEED = 6.28

# get the time step of the current world.
SIM_TIMESTEP = int(robot.getBasicTimeStep())


# Initialize Motors
leftMotor = robot.getDevice('left wheel motor')
rightMotor = robot.getDevice('right wheel motor')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))
leftMotor.setVelocity(0.0)
rightMotor.setVelocity(0.0)

# Initialize and Enable the Ground Sensors
gsr = [0, 0, 0]
ground_sensors = [robot.getDevice('gs0'), robot.getDevice('gs1'), robot.getDevice('gs2')]
for gs in ground_sensors:
    gs.enable(SIM_TIMESTEP)

# Allow sensors to properly initialize
for i in range(10): robot.step(SIM_TIMESTEP)  

vL = 0 # TODO: Initialize variable for left speed
vR = 0 # TODO: Initialize variable for right speed
elapsed_time = 0 
prev_elapsed_time = 0
sensor_time_elapsed = 0

def update_odometry():
    global pose_x, pose_y, pose_theta, prev_elapsed_time, elapsed_time
    
    elapsed_time += SIM_TIMESTEP / 1000.0
    delta_time = elapsed_time - prev_elapsed_time
    
    right_normalized_speed = (vR / MAX_SPEED) * EPUCK_MAX_WHEEL_SPEED
    left_normalized_speed = (vL / MAX_SPEED) * EPUCK_MAX_WHEEL_SPEED
    
    pose_theta += ((right_normalized_speed - left_normalized_speed) / EPUCK_AXLE_DIAMETER) * delta_time
    
    pose_x += math.cos(pose_theta) * ((left_normalized_speed + right_normalized_speed) / 2) * delta_time
    pose_y += math.sin(pose_theta) * ((left_normalized_speed + right_normalized_speed) / 2) * delta_time
    
    prev_elapsed_time = elapsed_time
# Main Control Loop:
state = 'speed_measurement'
while robot.step(SIM_TIMESTEP) != -1:
    
    for i, gs in enumerate(ground_sensors):
        gsr[i] = gs.getValue()
        
    center_sensor = gsr[1] < 800
    left_sensor = gsr[0] < 800
    right_sensor = gsr[2] < 800
    start_line = center_sensor and left_sensor and right_sensor
    
    if state == 'speed_measurement':
        if start_line:
            state = 'line_follower'
           
        vL = MAX_SPEED
        vR = MAX_SPEED
        
    elif state == 'line_follower':
    
        leftMax = leftMotor.getMaxVelocity()
        rightMax = rightMotor.getMaxVelocity() 

        if center_sensor: #go straight
            vL = leftMax*.25
            vR = rightMax*.25
       
        elif left_sensor:#move counterclockwise in place
            vL = -leftMax*0.15
            vR = rightMax*0.2
        elif right_sensor:
            vL = leftMax*0.2
            vR = -rightMax*0.15
        else:
            vL = -leftMax*0.15
            vR = rightMax*0.2
        
        update_odometry()
  
    #print(gsr) # TODO: Uncomment to see the ground sensor values!

        
    # Hints: 
    #
    # 1) Setting vL=MAX_SPEED and vR=-MAX_SPEED lets the robot turn
    # right on the spot. vL=MAX_SPEED and vR=0.5*MAX_SPEED lets the
    # robot drive a right curve.
    #
    # 2) If your robot "overshoots", turn slower.
    #
    # 3) Only set the wheel speeds once so that you can use the speed
    # that you calculated in your odometry calculation.
    #
    # 4) Disable all console output to simulate the robot superfast
    # and test the robustness of your approach.
    #

    # TODO: Insert Line Following Code Here  
    
    # TODO: Call update_odometry Here

    # Hints:
    #
    # 1) Divide vL/vR by MAX_SPEED to normalize, then multiply with
    # the robot's maximum speed in meters per second. 
    #
    # 2) SIM_TIMESTEP tells you the elapsed time per step. You need
    # to divide by 1000.0 to convert it to seconds
    #
    # 3) Do simple sanity checks. In the beginning, only one value
    # changes. Once you do a right turn, this value should be constant.
    #
    # 4) Focus on getting things generally right first, then worry
    # about calculating odometry in the world coordinate system of the
    # Webots simulator first (x points down, y points right)

    
    # TODO: Insert Loop Closure Code Here
    
    # Hints:
    #
    # 1) Set a flag whenever you encounter the line
    #
    # 2) Use the pose when you encounter the line last 
    # for best results
    if start_line:
        sensor_time_elapsed += SIM_TIMESTEP / 1000.0
        if sensor_time_elapsed >= .1:
            pose_x = 0
            pose_y = 0
            pose_theta = 0
    else:
        sensor_time_elapsed  = 0
    
    print("Current pose: [%5f, %5f, %5f]" % (pose_x, pose_y, pose_theta))
    leftMotor.setVelocity(vL)
    rightMotor.setVelocity(vR)