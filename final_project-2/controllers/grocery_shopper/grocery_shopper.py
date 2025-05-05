"""grocery controller."""

# HOW TO RUN

# INSTALLS:
# pip install ultralytics
# pip install scipy

# TO RUN:
# Normal with webots file - just press play if running from github repository

# ARM CONTROLLER
# W: forward 
# S: backward 
# A: left 
# D: right 
# E: up 
# Q: down 
# O: open grip 
# C: close grip 
# R: reset if getting unreachable warning 
# H: put arm in front of camera G: put arm in front of basket

# WEBOTS WORLD CHANGES
# Changed lidar angle so that it wasn't hitting the ground, rotated upwards.


# Apr 1, 2025
from controller import Robot, Motor, Camera, RangeFinder, Lidar, Keyboard # type: ignore 
import math
import cv2
from ultralytics import YOLO
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d # Uncomment if you want to use something else for finding the configuration space
import matplotlib.transforms as transforms
import heapq
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink
import ikpy.utils.plot as plot_utils
import gripper as grip

# our files
import map_with_lidar as lid
import rrt as rrt
import ik as ik
import cv as cv

model = YOLO('best.pt')
CONF = .25

#Initialization
print("=== Initializing Grocery Shopper...")
#Consts
MAX_SPEED = 7.0  # [rad/s]
MAX_SPEED_MS = 0.633 # [m/s]
AXLE_LENGTH = 0.4044 # m
MOTOR_LEFT = 10
MOTOR_RIGHT = 11
N_PARTS = 12
LIDAR_ANGLE_BINS = 667
LIDAR_SENSOR_MAX_RANGE = 5.5 # Meters
LIDAR_ANGLE_RANGE = math.radians(240)

# create the Robot instance.
robot = Robot()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# The Tiago robot has multiple motors, each identified by their names below
part_names = ("head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
              "arm_2_joint",  "arm_3_joint",  "arm_4_joint",      "arm_5_joint",
              "arm_6_joint",  "arm_7_joint",  "wheel_left_joint", "wheel_right_joint",
              "gripper_left_finger_joint","gripper_right_finger_joint")

# 

# All motors except the wheels are controlled by position control. The wheels
# are controlled by a velocity controller. We therefore set their position to infinite.
target_pos = (0.0, 0.0, 0.35, 0.07, 1.02, -3.16, 1.27, 1.32, 0.0, 1.41, 'inf', 'inf',0.045,0.045)

robot_parts={}
for i, part_name in enumerate(part_names):
    robot_parts[part_name]=robot.getDevice(part_name)
    robot_parts[part_name].setPosition(float(target_pos[i]))
    robot_parts[part_name].setVelocity(robot_parts[part_name].getMaxVelocity() / 2.0)

# Enable gripper encoders (position sensors)
left_gripper_enc=robot.getDevice("gripper_left_finger_joint_sensor")
right_gripper_enc=robot.getDevice("gripper_right_finger_joint_sensor")
left_gripper_enc.enable(timestep)
right_gripper_enc.enable(timestep)

# initialize the teleoperation class with a reference to the robot for IK
grippee = grip.Gripper(robot)

# Enable Range Finder
depth_cam = robot.getDevice('range-finder')
depth_cam.enable(timestep)
depth_W = depth_cam.getWidth()
depth_H = depth_cam.getHeight()

# Enable Camera
camera = robot.getDevice('camera')
camera.enable(timestep)
camera.recognitionEnable(timestep)

# Enable GPS and compass localization
gps = robot.getDevice("gps")
gps.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)

# Enable LiDAR
lidar = robot.getDevice('Hokuyo URG-04LX-UG01')
lidar.enable(timestep)
lidar.enablePointCloud()

# Enable display
display = robot.getDevice("display")

# Keyboard
keyboard = robot.getKeyboard()
keyboard.enable(timestep)

# Movement/ RRT*
curr_waypoint = 0
map_waypoints = []
world_waypoints = []
node_list = []
last_waypoints_angle = 0
ahead_goal_attempts = 0
valid_goal = True

# Avoiding Obstacles
avoiding_object = False
avoiding_object_angle = 0
elapsed_time = 0
avoiding_time = 10 # 10s for the robot to spin 360 degrees to capture more lidar data

startup_time = 0
startup_limit = 2

# Odometry
pose_x     = 0
pose_y     = 0
pose_theta = 0

vL = 0
vR = 0

# Lidar
lidar_sensor_readings = [] # List to hold sensor readings
lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE/2., +LIDAR_ANGLE_RANGE/2., LIDAR_ANGLE_BINS)
lidar_offsets = lidar_offsets[83:len(lidar_offsets)-83] # Only keep lidar readings not blocked by robot chassis
lidar_threshold = 1.2 # if object in front of robot is within this meter threshold
lidar_front_readings = []
lidar_center = 250
lidar_width = 75


# ------------------------------------------------------------------
# Helper Functions


gripper_status="closed"


# important variables
lidar_map = np.zeros(shape=[360,360])
filtered_lidar_map = np.zeros(shape=[360,360])
lidar_map_generated = False

# for resetting variables before running RRT*
def reset_variables():
    global map_waypoints, world_waypoints, node_list, last_waypoints_angle, valid_goal, ahead_goal_attempts, goal_point
    map_waypoints = []
    world_waypoints = []
    node_list = []
    last_waypoints_angle = pose_theta
    valid_goal = True
    ahead_goal_attempts = 0
    goal_point = (0, 0)

# Main Loop
while robot.step(timestep) != -1:
    ##########################################################################################
    # POSITIONING - ODOMETRY
    ##########################################################################################
    pose_x = gps.getValues()[0]
    pose_y = gps.getValues()[1]
    
    n = compass.getValues()
    rad = -((math.atan2(n[0], n[2]))-1.5708)
    pose_theta = rad
    world_theta = pose_theta + math.pi/2

    ##########################################################################################
    # LIDAR/MAPS
    ##########################################################################################

    lidar_sensor_readings = lidar.getRangeImage()
    lidar_sensor_readings = lidar_sensor_readings[83:len(lidar_sensor_readings)-83]
    lidar_front_readings = lidar_sensor_readings[lidar_center - lidar_width : lidar_center + lidar_width + 1]
    current_map_location = lid.globalcoords_to_map_coords(pose_x, pose_y)

    # make/update the lidar map on every robot step
    lidar_map_generated = lid.make_lidar_map(pose_x, pose_y, pose_theta, lidar_map, lidar_sensor_readings, display)

    # allowing some time after startup for lidar to load, leads to smoother startup
    startup_time += timestep / 1000.0
    if startup_time < startup_limit:
        vL, vR = (0, 0)
        continue
    
    ##########################################################################################
    # teleoperation - moving the arm
    ##########################################################################################
    key = keyboard.getKey()
    if key == ord('S'):  # Move backwardf
        grippee.tele_increment([-1, 0, 0])
    elif key == ord('W'):  # Move forward
        grippee.tele_increment([1, 0, 0])
    elif key == ord('A'):  # Move left
        grippee.tele_increment([0, -1, 0])
    elif key == ord('D'):  # Move right
        grippee.tele_increment([0, 1, 0])
    elif key == ord('E'):  # Move up
        grippee.tele_increment([0, 0, 1])
    elif key == ord('Q'):  # Move down
        grippee.tele_increment([0, 0, -1])
    elif key == ord('O'):  # Open grip
        grippee.openGrip()
    elif key == ord('C'):  # Close grip
        grippee.closeGrip()
    elif key == ord('G'):  # Go to basket
        grippee.move_to_basket()
    elif key == ord('H'):  # Go to viewport
        grippee.move_arm_to_position([0, -.3, -.25])
    elif key == ord('R'):  # Reset motors
        grippee.reset_motors()

    ##########################################################################################
    # RRT
    ##########################################################################################

    # if the current waypoint is at the end of the waypoints list or the waypoints list is not populated, then filter the lidar map and run RRT*
    if (curr_waypoint == len(world_waypoints)-1 or len(map_waypoints) == 0) and lidar_map_generated:
        reset_variables()
        print("filtering...")

        # filter the lidar map to feed into RRT*
        filtered_lidar_map = lid.filter_lidar_map(lidar_map)
        filtered_lidar_map = lid.expand_pixels(filtered_lidar_map, box_size=5)
        lid.display_map(display, filtered_lidar_map)

        current_map_position = lid.globalcoords_to_map_coords(pose_x, pose_y)
       
        # update map based on new lidar map
        frontiers, unknown, explored, obstacles = rrt.map_update(filtered_lidar_map)
        bounds = np.array([[0,360],[0,360]])

        # while there RRT* has not created a successful path to goal
        while len(map_waypoints) == 0 and len(frontiers) != 0:
            print('entered into the rrt while loop')
            print('length of frontiers: ', len(frontiers))
            print(frontiers)

            # attempt to generate a random frontier in front of the robot
            # after 5 failed attempts, of trying to get a frontier in front, then get any random frontier as next goal point
            if valid_goal == True and ahead_goal_attempts < 5:
                result = rrt.get_random_frontier_vertex_ahead(current_map_position[0], current_map_position[1], last_waypoints_angle)
                if result is not None:
                    goal_point, valid_goal = result
                print("FIRST TRIED GOAL POINT: ", goal_point)
                print("VALID GOAL BOOL: ", valid_goal)
                ahead_goal_attempts += 1

            else:
                goal_point = rrt.get_random_frontier_vertex()
                print('ANOTHER TRIED GOAL POINT: ', goal_point)

            # run RRT* and visualize
            node_list, map_waypoints = rrt.rrt_star(filtered_lidar_map, bounds, rrt.obstacles, rrt.point_is_valid, current_map_position, goal_point, 250, 30)
            if len(node_list) > 0:
                rrt.visualize_2D_graph(bounds, rrt.obstacles, node_list, goal_point, 'robot_rrt_star_run.png')

        # after the map waypoints have been generated, convert into world waypoints to feed to robot
        if map_waypoints is not None and len(map_waypoints) > 0:
            world_waypoints = [lid.map_coords_to_global_coords(pt[0], pt[1]) for pt in map_waypoints]
        else:
            print("map_waypoints is None!")

        curr_waypoint = 0
        elapsed_time = 0
    
    print('pos: ', pose_x, pose_y, world_theta)
    print('avoiding object val: ', avoiding_object)
    
    valid_goal = True

    ##########################################################################################
    # OBSTACLE AVOIDANCE 
    ##########################################################################################
    if (len(lidar_front_readings) > 0):
        print('lowest front lidar reading: ', np.min(np.array(lidar_front_readings)))

    # if any of the front lidar readings are below the threshold, then avoid the object by resetting variables and re-running RRT*
    if np.any(np.array(lidar_front_readings) < lidar_threshold) and avoiding_object == False:
        reset_variables()
        avoiding_object = True
        avoiding_object_angle = world_theta
        print('AVOIDING OBJECT!')

    # update avoid object condition
    if avoiding_object == True:
            elapsed_time += timestep / 1000.0
            print(elapsed_time)
            if elapsed_time >= avoiding_time or ((world_theta - avoiding_object_angle + math.pi) % (2 * math.pi) - math.pi) > np.radians(30) or curr_waypoint > 0:
                avoiding_object = False

    # find the last waypoint direction to determine the direction that the next RRT* goal should be in
    if len(map_waypoints) > 1:
        last_waypoints_angle = rrt.get_last_waypoint_direction(map_waypoints[-1], map_waypoints[0])
    prev_wp = curr_waypoint

    # navigate to the current waypoint
    vels, curr_waypoint = ik.nav_to_waypoint(world_waypoints, curr_waypoint, pose_x, pose_y, world_theta)
    
    print("current waypoint", curr_waypoint)
    print("prev_wp waypoint", prev_wp)
    if curr_waypoint != prev_wp:        
        robot_parts["wheel_left_joint"].setVelocity(0.0)
        robot_parts["wheel_right_joint"].setVelocity(0.0)
        print("running cv")
        cv.run_cv(camera, depth_cam)

      
        robot.step(3 * timestep) 

    vL, vR = vels
    
    ##########################################################################################
    # MOVING
    ##########################################################################################
    
    robot_parts["wheel_left_joint"].setVelocity(vL)
    robot_parts["wheel_right_joint"].setVelocity(vR)
