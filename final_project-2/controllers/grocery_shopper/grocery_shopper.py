"""grocery controller."""

# Apr 1, 2025
from controller import Robot, Motor, Camera, RangeFinder, Lidar, Keyboard # type: ignore 
import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d # Uncomment if you want to use something else for finding the configuration space
import matplotlib.transforms as transforms
import heapq
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink
import ikpy.utils.plot as plot_utils
import time

# our files
import map_with_lidar as lid
import rrt as rrt
import ik as ik



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

# RRT and Path Planning
curr_waypoint = 0
elapsed_time = 0
forward_state = 0
map_waypoints = []
world_waypoints = []
forward_state = 0
node_list = []
last_waypoints_angle = 0
ahead_goal_attempts = 0
valid_goal = True

# Odometry
pose_x     = 0
pose_y     = 0
pose_theta = 0

vL = 0
vR = 0

lidar_sensor_readings = [] # List to hold sensor readings
lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE/2., +LIDAR_ANGLE_RANGE/2., LIDAR_ANGLE_BINS)
lidar_offsets = lidar_offsets[83:len(lidar_offsets)-83] # Only keep lidar readings not blocked by robot chassis
lidar_threshold = 0.6 # if object in front of robot is within this meter threshold
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

def reset_variables():
    global map_waypoints, world_waypoints, node_list, last_waypoints_angle, valid_goal, ahead_goal_attempts
    map_waypoints = []
    world_waypoints = []
    node_list = []
    last_waypoints_angle = pose_theta
    valid_goal = True
    ahead_goal_attempts = 0

# Main Loop
while robot.step(timestep) != -1:
    
    ##########################################################################################
    # POSITIONING - ODOMETRY
    ##########################################################################################
    pose_x = gps.getValues()[0]
    pose_y = gps.getValues()[1]
    

    elapsed_time += timestep / 1000.0
    delta_time = timestep / 1000.0
    
    n = compass.getValues()
    rad = -((math.atan2(n[0], n[2]))-1.5708)
    pose_theta = rad
    world_theta = pose_theta + math.pi/2

    lidar_sensor_readings = lidar.getRangeImage()
    lidar_sensor_readings = lidar_sensor_readings[83:len(lidar_sensor_readings)-83]
    lidar_front_readings = lidar_sensor_readings[lidar_center - lidar_width : lidar_center + lidar_width + 1]
    current_map_location = lid.globalcoords_to_map_coords(pose_x, pose_y)
    
    ##########################################################################################
    # LIDAR/MAPS
    ##########################################################################################

    # make/update the lidar map on every robot step
    lidar_map_generated = lid.make_lidar_map(pose_x, pose_y, pose_theta, lidar_map, lidar_sensor_readings, display)
    
    key = keyboard.getKey()

    if (curr_waypoint == len(world_waypoints)-1 or len(map_waypoints) == 0) and lidar_map_generated:
        reset_variables()
        print("filtering...")
        filtered_lidar_map = lid.filter_lidar_map(lidar_map)
        filtered_lidar_map = lid.expand_pixels(filtered_lidar_map, box_size=10)
        lid.display_map(display, filtered_lidar_map)

        current_map_position = lid.globalcoords_to_map_coords(pose_x, pose_y)
        frontiers, unknown, explored, obstacles = rrt.map_update(filtered_lidar_map)
        bounds = np.array([[0,360],[0,360]])

        while len(map_waypoints) == 0 and len(frontiers) != 0:
            if valid_goal == True:
                goal_point, valid_goal = rrt.get_random_frontier_vertex_ahead(current_map_position[0], current_map_position[1], last_waypoints_angle)
            else:
                goal_point = rrt.get_random_frontier_vertex()

            node_list, map_waypoints = rrt.rrt_star(filtered_lidar_map, bounds, rrt.obstacles, rrt.point_is_valid, current_map_position, goal_point, 500, 30)
            if len(node_list) > 0:
                rrt.visualize_2D_graph(bounds, rrt.obstacles, node_list, goal_point, 'robot_rrt_star_run.png')

        if map_waypoints is not None and len(map_waypoints) > 0:
            world_waypoints = [lid.map_coords_to_global_coords(pt[0], pt[1]) for pt in map_waypoints]
        else:
            print("map_waypoints is None!")

        curr_waypoint = 0
        elapsed_time = 0
    
    print('pos: ', pose_x, pose_y, world_theta)
    
    valid_goal = True

    if (len(lidar_front_readings) > 0):
        print('lowest front lidar reading: ', np.min(np.array(lidar_front_readings)))
    if np.any(np.array(lidar_front_readings) < lidar_threshold):
        reset_variables()
        print('AVOIDING OBJECT!')

    if len(map_waypoints) > 1:
        last_waypoints_angle = rrt.get_last_waypoint_direction(map_waypoints[-1], map_waypoints[-2])
    vels, curr_waypoint = ik.nav_to_waypoint(world_waypoints, curr_waypoint, pose_x, pose_y, world_theta)

    vL = vels[0]
    vR = vels[1]
    
    ##########################################################################################
    # MOVING
    ##########################################################################################
    
    robot_parts["wheel_left_joint"].setVelocity(vL)
    robot_parts["wheel_right_joint"].setVelocity(vR)
    
    
    ##########################################################################################
    # GRABBING ARM
    ##########################################################################################
    if(gripper_status=="open"):
        # Close gripper, note that this takes multiple time steps...
        robot_parts["gripper_left_finger_joint"].setPosition(0)
        robot_parts["gripper_right_finger_joint"].setPosition(0)
        if right_gripper_enc.getValue()<=0.005:
            gripper_status="closed"
    else:
        # Open gripper
        robot_parts["gripper_left_finger_joint"].setPosition(0.045)
        robot_parts["gripper_right_finger_joint"].setPosition(0.045)
        if left_gripper_enc.getValue()>=0.044:
            gripper_status="open"
