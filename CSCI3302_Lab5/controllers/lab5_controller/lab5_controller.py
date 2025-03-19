"""lab5 controller."""
from controller import Robot, Motor, Camera, RangeFinder, Lidar, Keyboard
import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d # Uncomment if you want to use something else for finding the configuration space
import matplotlib.transforms as transforms
import heapq

MAX_SPEED = 7.0  # [rad/s]
MAX_SPEED_MS = 0.633 # [m/s]
AXLE_LENGTH = 0.4044 # m
MOTOR_LEFT = 10
MOTOR_RIGHT = 11
N_PARTS = 12

LIDAR_ANGLE_BINS = 667
LIDAR_SENSOR_MAX_RANGE = 2.75 # Meters
LIDAR_ANGLE_RANGE = math.radians(240)

##### vvv [Begin] Do Not Modify vvv #####

# create the Robot instance.
robot = Robot()
# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# The Tiago robot has multiple motors, each identified by their names below
part_names = ("head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
              "arm_2_joint",  "arm_3_joint",  "arm_4_joint",      "arm_5_joint",
              "arm_6_joint",  "arm_7_joint",  "wheel_left_joint", "wheel_right_joint")

# All motors except the wheels are controlled by position control. The wheels
# are controlled by a velocity controller. We therefore set their position to infinite.
target_pos = (0.0, 0.0, 0.09, 0.07, 1.02, -3.16, 1.27, 1.32, 0.0, 1.41, 'inf', 'inf')
robot_parts=[]

for i in range(N_PARTS):
    robot_parts.append(robot.getDevice(part_names[i]))
    robot_parts[i].setPosition(float(target_pos[i]))
    robot_parts[i].setVelocity(robot_parts[i].getMaxVelocity() / 2.0)

# The Tiago robot has a couple more sensors than the e-Puck
# Some of them are mentioned below. We will use its LiDAR for Lab 5

range = robot.getDevice('range-finder')
range.enable(timestep)
camera = robot.getDevice('camera')
camera.enable(timestep)
camera.recognitionEnable(timestep)
lidar = robot.getDevice('Hokuyo URG-04LX-UG01')
lidar.enable(timestep)
lidar.enablePointCloud()

# We are using a GPS and compass to disentangle mapping and localization
gps = robot.getDevice("gps")
gps.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)

# We are using a keyboard to remote control the robot
keyboard = robot.getKeyboard()
keyboard.enable(timestep)

# The display is used to display the map. We are using 360x360 pixels to
# map the 12x12m2 apartment
display = robot.getDevice("display")

# Odometry
pose_x     = 0
pose_y     = 0
pose_theta = 0

vL = 0
vR = 0

lidar_sensor_readings = [] # List to hold sensor readings
lidar_offsets = np.linspace(-LIDAR_ANGLE_RANGE/2., +LIDAR_ANGLE_RANGE/2., LIDAR_ANGLE_BINS)
lidar_offsets = lidar_offsets[83:len(lidar_offsets)-83] # Only keep lidar readings not blocked by robot chassis

# map = None
##### ^^^ [End] Do Not Modify ^^^ #####

##################### IMPORTANT #####################
# Set the mode here. Please change to 'autonomous' before submission
# mode = 'manual' # Part 1.1: manual mode
# mode = 'planner'
mode = 'autonomous'
# mode = 'picknplace'

def dijkstra(map, start, end):
    
    rows, cols = map.shape
    open_set = []
    heapq.heappush(open_set, (0, start))  

    came_from = {}
    cost_so_far = {start: 0}

    while open_set:
        current_cost, current = heapq.heappop(open_set)

        if current == end:
            #reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path  
            
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (1, 1), (-1, -1), (-1, 1), (1, -1)]:
            neighbor = (current[0] + dx, current[1] + dy)

            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and map[neighbor] == 0:
                new_cost = cost_so_far[current] + 1  

                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost  
                    heapq.heappush(open_set, (priority, neighbor))
                    came_from[neighbor] = current  

    return []

def map_to_world(map_x, map_y):
    """ Converts map indices to Webots world coordinates (meters) """
    world_x = (map_x / 360) * 12 - 12
    world_y = (map_y / 360) * 12 - 12
    return (world_x, world_y)

def world_to_map(world_x, world_y):
    map_x = int(((world_x + 12)/12)* 360)
    map_y = int(((world_y + 12)/12)* 360)
    return (map_x, map_y)

###################
#
# Planner
#
###################
if mode == 'planner':
    # Part 2.3: Provide start and end in world coordinate frame and convert it to map's frame
    start_w = None # (Pose_X, Pose_Y) in meters
    end_w = None # (Pose_X, Pose_Y) in meters

    # Convert the start_w and end_w from the webots coordinate frame into the map frame
    start = None # (x, y) in 360x360 map
    end = None # (x, y) in 360x360 map

    # Part 2.3: Implement A* or Dijkstra's Algorithm to find a path
    # def path_planner(map, start, end):
    #     '''
    #     :param map: A 2D numpy array of size 360x360 representing the world's cspace with 0 as free space and 1 as obstacle
    #     :param start: A tuple of indices representing the start cell in the map
    #     :param end: A tuple of indices representing the end cell in the map
    #     :return: A list of tuples as a path from the given start to the given end in the given maze
    #     '''
    #     rows, cols = map.shape
    #     open_set = []
    #     heapq.heappush(open_set, (0, start))  

    #     came_from = {}
    #     cost_so_far = {start: 0}

    #     while open_set:
    #         current_cost, current = heapq.heappop(open_set)

    #         if current == end:
    #             #reconstruct path
    #             path = []
    #             while current in came_from:
    #                 path.append(current)
    #                 current = came_from[current]
    #             path.append(start)
    #             path.reverse()
    #             return path  
                
    #         for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (1, 1), (-1, -1), (-1, 1), (1, -1)]:
    #             neighbor = (current[0] + dx, current[1] + dy)

    #             if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and map[neighbor] == 0:
    #                 new_cost = cost_so_far[current] + 1  

    #                 if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
    #                     cost_so_far[neighbor] = new_cost
    #                     priority = new_cost  
    #                     heapq.heappush(open_set, (priority, neighbor))
    #                     came_from[neighbor] = current  

    #     return []
        # pass

    # Part 2.1: Load map (map.npy) from disk and visualize it
    filtered_map = np.load("../../maps/mapv1.npy")
    plt.imshow(filtered_map, cmap='gray', origin="lower")
    # plt.imshow(np.fliplr(filtered_map), cmap='gray', origin="upper")
    # plt.xticks(rotation=90)
    # plt.yticks(rotation=90)
    
    # plt.show()
    

    # Part 2.2: Compute an approximation of the “configuration space”
    # Convolve the map with a quadratic kernel of ones to approximate the configuration space
    kernel = np.ones((10, 10))  # Define a 3x3 kernel of ones
    map = convolve2d(filtered_map, kernel, mode='same', boundary='wrap')  # Convolve the map with the kernel
    map = map > 0  # Convert the convolved map to binary, where 1 represents an obstacle or its configuration space
    plt.imshow(map, cmap='gray', origin="lower")
    plt.show()
    
    # Part 2.3 continuation: Call path_planner
    

    # Part 2.4: Turn paths into waypoints and save on disk as path.npy and visualize it
    waypoints = []

######################
#
# Map Initialization
#
######################

# Part 1.2: Map Initialization

# Initialize your map data structure here as a 2D floating point array
map_data = np.zeros(shape=[360,360])

waypoints = []

if mode == 'autonomous':
    # Part 3.1: Load path from disk and visualize it
    waypoints = [] # Replace with code to load your path
    filtered_map = np.load("../../maps/mapv1.npy")
    kernel = np.ones((17, 17)) 
    map_cspace = convolve2d(filtered_map, kernel, mode='same', boundary='wrap')
    map_cspace = map_cspace > 0
    # start = (316,300)
    # start = (200, 200)
    start = world_to_map(pose_x, pose_y)
    end = (100,123)

    path = dijkstra(map_cspace, start, end)
    waypoints = [map_to_world(x, y) for (x, y) in path]

    filtered_waypoints = [i for j, i in enumerate(waypoints) if j % 15 == 0]
    filtered_waypoints.append(waypoints[len(waypoints)-1])
    filtered_waypoints = [(y, x) for (x, y) in filtered_waypoints]


    np.save("../../maps/path.npy", waypoints)
    map_display = map_cspace.copy()
    if path:
        print("Path saved!")
    else:
        print("No valid path found")
    for (x, y) in path:
        map_display[x, y] = 0.5  
    
    plt.figure(figsize=(6,6))
    plt.title("Path Visualization")
    plt.imshow(map_display, cmap='gray', origin="lower")
    
    plt.show()

state = 0 # use this to iterate through your path
elapsed_time = 0

if mode == 'picknplace':
    # Part 4: Use the function calls from lab5_joints using the comments provided there
    ## use path_planning to generate paths
    ## do not change start_ws and end_ws below
    start_ws = [(3.7, 5.7)]
    end_ws = [(10.0, 9.3)]
    pass

while robot.step(timestep) != -1 and mode != 'planner':

    ###################
    #
    # Mapping
    #
    ###################

    ################ v [Begin] Do not modify v ##################
    # Ground truth pose
    pose_x = gps.getValues()[0]
    pose_y = gps.getValues()[1]
    
    n = compass.getValues()
    rad = -((math.atan2(n[0], n[2]))-1.5708)
    pose_theta = rad

    lidar_sensor_readings = lidar.getRangeImage()
    lidar_sensor_readings = lidar_sensor_readings[83:len(lidar_sensor_readings)-83]
    
    # print(lidar_sensor_readings)
    for i, rho in enumerate(lidar_sensor_readings):
        alpha = lidar_offsets[i]

        if rho > LIDAR_SENSOR_MAX_RANGE:
            continue

        # The Webots coordinate system doesn't match the robot-centric axes we're used to
        rx = math.cos(alpha)*rho
        ry = -math.sin(alpha)*rho

        t = pose_theta + np.pi/2.
        # Convert detection from robot coordinates into world coordinates
        wx =  math.cos(t)*rx - math.sin(t)*ry + pose_x
        wy =  math.sin(t)*rx + math.cos(t)*ry + pose_y
        
        # print(rx, ry)÷s

        ################ ^ [End] Do not modify ^ ##################

        #print("Rho: %f Alpha: %f rx: %f ry: %f wx: %f wy: %f" % (rho,alpha,rx,ry,wx,wy))
        if wx >= 12:
            wx = 11.999
        if wy >= 12:
            wy = 11.999
        if rho < LIDAR_SENSOR_MAX_RANGE:
            # Part 1.3: visualize map gray values.
            
            #try catch for bounds just in case
            try:
                grey_val = min(map_data[359-abs(int(wx*30))][abs(int(wy*30))] + .005, 1.0)
                map_data[359-abs(int(wx*30))][abs(int(wy*30))] = grey_val
                # You will eventually REPLACE the following lines with a more robust version of the map
                # with a grayscale drawing containing more levels than just 0 and 1.
                
                color = (grey_val*256**2+grey_val*256+grey_val)*255
                display.setColor(int(color))
                # display.setColor((map[360-abs(int(wx*30))][abs(int(wy*30))]*256**2+map[360-abs(int(wx*30))][abs(int(wy*30))]*256+map[360-abs(int(wx*30))][abs(int(wy*30))])*255)
                display.drawPixel(360-abs(int(wx*30)),abs(int(wy*30)))
            except:
                pass
                
            # print((360-abs(int(wx*30)),abs(int(wy*30))))
            # print(map[360-abs(int(wx*30))][abs(int(wy*30))]) 
            

    # Draw the robot's current pose on the 360x360 display
    display.setColor(int(0xFF0000))
    display.drawPixel(360-abs(int(pose_x*30)), abs(int(pose_y*30)))

    ###################
    #
    # Controller
    #
    ###################
    if mode == 'manual':
        key = keyboard.getKey()
        while(keyboard.getKey() != -1): pass
        if key == keyboard.LEFT :
            vL = -MAX_SPEED
            vR = MAX_SPEED
        elif key == keyboard.RIGHT:
            vL = MAX_SPEED
            vR = -MAX_SPEED
        elif key == keyboard.UP:
            vL = MAX_SPEED
            vR = MAX_SPEED
        elif key == keyboard.DOWN:
            vL = -MAX_SPEED
            vR = -MAX_SPEED
        elif key == ord(' '):
            vL = 0
            vR = 0
        elif key == ord('S'):
            # Part 1.4: Filter map and save to filesystem
            filtered_map = np.where(map_data > .8, 1, 0)
            # print(filtered_map.shape)
            np.save("../../maps/mapv2.npy", filtered_map)
            np.save("../../maps/maprawv2.npy", map_data)
            print("Map file saved")
        elif key == ord('L'):
            # You will not use this portion in Part 1 but here's an example for loading saved a numpy array
            filtered_map = np.load("../../maps/mapv1.npy")
            filtered_map = np.load("../../maps/maprawv1.npy")
            print(filtered_map)
            print("Map loaded")
        else: # slow down
            vL *= 0.75
            vR *= 0.75
    else: # not manual mode
        # Part 3.2: Feedback controller
        #STEP 1: Calculate the error
        # rho = 0
        # alpha = 0

        #STEP 2: Controller
        # dX = 0
        # dTheta = 0

        #STEP 3: Compute wheelspeeds
        # vL = 0
        # vR = 0

        def inverse_wheel_kinematics(distance, delta_theta, delta_time=timestep / 1000.0, axle_diameter=AXLE_LENGTH):
            """takes in the amount to travel then gives the rotations we need"""
            if delta_time == 0: return 0, 0
            
            # reversed the equations we had from last lab 2 for odometry
            v_linear = distance / delta_time  
            vR = (delta_theta * axle_diameter) / (2 * delta_time) + v_linear
            vL = v_linear - (delta_theta * axle_diameter) / (2 * delta_time)
            return vL, vR

        def turn_to_goal(ang_to_goal: float, is_proportional=True) -> tuple:
            """ takes in the angle and turns if not facing -> returns true if it is facing the goal otherwise returns false"""
            
            # global leftMotor, rightMotor, leftMax, rightMax
            
            # tuning variables r here
            err_margin = .01
            turn_speed = .2 # default value will change this
            min_turn_speed = .01
            max_turn_speed = .25
            portional_gain = .3
            
            # this part does makes it turn faster the further it is
            if is_proportional:
                turn_speed = abs(portional_gain * ang_to_goal)
            
            if turn_speed > max_turn_speed:
                turn_speed = max_turn_speed
                
            if turn_speed < min_turn_speed:
                turn_speed = min_turn_speed

            
            if not (ang_to_goal < err_margin and ang_to_goal > -err_margin):
                if ang_to_goal > 0:
                    return (-MAX_SPEED * turn_speed, MAX_SPEED * turn_speed)
                return (MAX_SPEED * turn_speed, -MAX_SPEED * turn_speed)
                
            return None


        # pose_x = gps.getValues()[0]
        # pose_y = gps.getValues()[1]
        # pose_theta = np.arctan2(compass.getValues()[0], compass.getValues()[1])
        
        # get the times:
        elapsed_time += timestep / 1000.0
        delta_time = timestep / 1000.0
        
        # TODO: controller / calculations
        
        # euclidian distance
        # goal_pos = (-0.19, 0.125162, 0)
        # filtered_waypoints = waypoints[3::4]
        # goal_pos = waypoints[state]
        goal_pos = filtered_waypoints[state]
        # print('WAYPOINTS', waypoints)
        print('FILTERED WAYPOINTS', filtered_waypoints)
        euc_dis = math.pow(goal_pos[0] - pose_x, 2)
        euc_dis += math.pow(goal_pos[1] - pose_y, 2)
        euc_dis = math.sqrt(euc_dis)
        
        # calculate the angle to goal
        ang_to_goal = math.atan2(goal_pos[1] - pose_y, goal_pos[0] - pose_x)
        ang_to_goal = (ang_to_goal - pose_theta + math.pi) % (2 * math.pi) - math.pi
        # heading_to_goal_heading = goal_pos[2] - pose_theta
        # heading_to_goal_heading = (goal_pos[2] - pose_theta + math.pi) % (2 * math.pi) - math.pi

        # if is_proportional_feedback_controller_state:
            
        # tuning vars:
        forward_err = .01
        rot_err = .01
        forward_gain = 5
        rot_gain = .1
        
        R_dis, L_dis = inverse_wheel_kinematics(euc_dis, ang_to_goal)
        print('R AND L DISTANCE: ', R_dis, L_dis)
        wheel_rot = R_dis - L_dis # right wheel minus left gives positive theta rot
        
        if not (wheel_rot < rot_err and wheel_rot > -rot_err):
            if wheel_rot > 0:
                res = (-wheel_rot * rot_gain, wheel_rot * rot_gain)
            else:
                res = (wheel_rot * rot_gain, -wheel_rot * rot_gain)
            
        else:
            res = (R_dis * forward_gain, L_dis * forward_gain)
            
        # set bounds
        if res[0] > .5:
            # res[0] = .5
            res = (.5, res[1])
        if res[1] > .5:
            # res[1] = .5
            res = (res[0], .5)

        if res[0] < .001 and res[1] < .001:
            state += 1

        # JUST MAKING THE CONTROLLER ALWAYS PROPORTIONAL
        # else:
        #     match state:
        #         case 0:
        #             res = turn_to_goal(ang_to_goal, is_proportional_controller)
        #             if res is None:
        #                 res = reach_position(euc_dis, is_proportional_controller)
        #                 if res is None:
        #                     state += 1
        #                     res = (0, 0)
                        
        #         case 1:
        #             res = turn_to_goal(heading_to_goal_heading, is_proportional_controller)
        #             if res is None:
        #                 res = (0, 0)
        #                 state = 0
        #                 index += 1
                    
        #         case _:
        #             res = (0, 0)
                    
        # WAS HERE BEFORE
        # leftMotor.setVelocity(res[0])
        # rightMotor.setVelocity(res[1])

        vL = res[0]
        vR = res[1]
        
        # exit condition here so that it ends
        if state >= len(waypoints):
            # leftMotor.setVelocity(0)
            # rightMotor.setVelocity(0)
            vL = 0
            vR = 0
            # exit(0)
        
        
        #############################################################
        # moving and printing stuff out
        #############################################################
        
        print("Current pose: [%5f, %5f, %5f]" % (pose_x, pose_y, pose_theta))
        print("euc distance: ", euc_dis)
        print("angle_to_goal: ", ang_to_goal)
        # print("heading_to_goal: ", heading_to_goal_heading)
        print(state)

        # Normalize wheelspeed
        # (Keep the wheel speeds a bit less than the actual platform MAX_SPEED to minimize jerk)


    # Odometry code. Don't change vL or vR speeds after this line.
    # We are using GPS and compass for this lab to get a better pose but this is how you'll do the odometry
    pose_x += (vL+vR)/2/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0*math.cos(pose_theta)
    pose_y -= (vL+vR)/2/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0*math.sin(pose_theta)
    pose_theta += (vR-vL)/AXLE_LENGTH/MAX_SPEED*MAX_SPEED_MS*timestep/1000.0

    # print("X: %f Z: %f Theta: %f" % (pose_x, pose_y, pose_theta))

    # Actuator commands
    robot_parts[MOTOR_LEFT].setVelocity(vL)
    robot_parts[MOTOR_RIGHT].setVelocity(vR)
    
while robot.step(timestep) != -1:
    # there is a bug where webots have to be restarted if the controller exits on Windows
    # this is to keep the controller running
    pass
