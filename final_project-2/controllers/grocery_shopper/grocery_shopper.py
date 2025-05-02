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
import gripper as grip

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
target_pos = (0.0, 0.0, 0, 0.07, 1.02, 0, 1.27, 1.32, 0.0, 1.41, 'inf', 'inf',0.045,0.045)

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

grippee = grip.Gripper(robot)

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

# Enable range
# range_finder = robot.getDevice('range-finder')
# range_finder.enable(timestep)

# Enable display
display = robot.getDevice("display")

# Keyboard
keyboard = robot.getKeyboard()
keyboard.enable(timestep)

curr_waypoint = 0
elapsed_time = 0
prev_time = 0
forward_state = 0
rrt_state = 'goal'
map_waypoints = []
world_waypoints = []
forward_state = 0

# Odometry
pose_x     = 0
pose_y     = 0
pose_theta = 0

vL = 0
vR = 0

lidar_sensor_readings = [] # List to hold sensor readings
lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE/2., +LIDAR_ANGLE_RANGE/2., LIDAR_ANGLE_BINS)
lidar_offsets = lidar_offsets[83:len(lidar_offsets)-83] # Only keep lidar readings not blocked by robot chassis



# ------------------------------------------------------------------
# Helper Functions


gripper_status="closed"


# important variables
lidar_map = np.zeros(shape=[360,360])
filtered_lidar_map = np.zeros(shape=[360,360])
current_map_location = ()


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

    lidar_sensor_readings = lidar.getRangeImage()
    lidar_sensor_readings = lidar_sensor_readings[83:len(lidar_sensor_readings)-83]
    current_map_location = lid.globalcoords_to_map_coords(pose_x, pose_y)
    
    ##########################################################################################
    # LIDAR/MAPS
    ##########################################################################################

    # make/update the lidar map on every robot step
    lid.make_lidar_map(pose_x, pose_y, pose_theta, lidar_map, lidar_sensor_readings, display)
    
    
    key = keyboard.getKey()
    if key == ord('S'):
        grippee.tele_increment([-1, 0, 0])
    elif key == ord('W'):
        grippee.tele_increment([1, 0, 0])
    elif key ==ord("A"):
        grippee.tele_increment([0, -1, 0])
    elif key == ord("D"):
        grippee.tele_increment([0, 1, 0])
    elif key == ord("E"):
        grippee.tele_increment([0, 0, 1])
    elif key == ord("Q"):
        grippee.tele_increment([0, 0, -1])
    elif key == ord("O"):
        grippee.openGrip()
    elif key == ord("C"):
        grippee.closeGrip()
    elif key == ord("G"):
        # go to basked
        grippee.move_to_basket()
    elif key == ord("H"):
        # go to viewport
        grippee.move_arm_to_position([0, -.3, -.25])
    else: pass
    
    # if key == ord('S'):
    #     print("filtering...")
    #     filtered_lidar_map = lid.filter_lidar_map(lidar_map)
    #     filtered_lidar_map = lid.expand_pixels(filtered_lidar_map, box_size=5)
    #     lid.display_map(display, filtered_lidar_map)

    #     current_map_position = lid.globalcoords_to_map_coords(pose_x, pose_y)
    #     # update based on new map
    #     frontiers, unknown, explored, obstacles = rrt.map_update(filtered_lidar_map)
    #     # print(frontiers)
    #     goal_point = rrt.get_random_frontier_vertex()
    #     bounds = np.array([[0,360],[0,360]])
    #     node_list, map_waypoints = rrt.rrt_star(filtered_lidar_map, bounds, rrt.obstacles, rrt.point_is_valid, current_map_position, goal_point, 200, 30)
    #     print(node_list)
    #     print(map_waypoints)
    #     rrt.visualize_2D_graph(bounds, rrt.obstacles, node_list, goal_point, 'robot_rrt_star_run.png')

    #     if map_waypoints is not None:
    #         world_waypoints = [lid.map_coords_to_global_coords(pt[0], pt[1]) for pt in map_waypoints]
    #     else:
    #         print("map_waypoints is None!")
    #         world_waypoints = []

    #     # world_waypoints = [lid.map_coords_to_global_coords(pt[0], pt[1]) for pt in map_waypoints]
    #     curr_waypoint = 0

    # vels = ik.nav_to_waypoint(world_waypoints, curr_waypoint, pose_x, pose_y, pose_theta)
    # vL = vels[0]
    # vR = vels[1]

    ##########################################################################################
    # MOVING
    ##########################################################################################
    
    robot_parts["wheel_left_joint"].setVelocity(vL)
    robot_parts["wheel_right_joint"].setVelocity(vR)
