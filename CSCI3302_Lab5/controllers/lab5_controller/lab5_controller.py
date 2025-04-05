"""lab5 controller."""
from controller import Robot, Motor, Camera, RangeFinder, Lidar, Keyboard
import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d # Uncomment if you want to use something else for finding the configuration space
import matplotlib.transforms as transforms
import heapq
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink
import ikpy.utils.plot as plot_utils

MAX_SPEED = 7.0  # [rad/s]
MAX_SPEED_MS = 0.633 # [m/s]
AXLE_LENGTH = 0.4044 # m
MOTOR_LEFT = 10
MOTOR_RIGHT = 11
N_PARTS = 12

LIDAR_ANGLE_BINS = 667
LIDAR_SENSOR_MAX_RANGE = 2.75 # Meters
LIDAR_ANGLE_RANGE = math.radians(240)



ikCalculated = False
# moveArmRan = False
objectSpotted = False
objectReached = False
objectGrabbed = False
retracing = False

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
target_pos = (-0.2, 0.0, 0.09, 0.07, 1.02, -3.16, 1.27, 1.32, 0.0, 1.41, 'inf', 'inf')
robot_parts=[]
# robot.getDevice("gripper_right_finger_joint").setPosition(0.045)
# robot.getDevice("gripper_left_finger_joint").setPosition(0.045)
for i in range(N_PARTS):
    robot_parts.append(robot.getDevice(part_names[i]))
    robot_parts[i].setPosition(float(target_pos[i]))
   
    robot_parts[i].setVelocity(robot_parts[i].getMaxVelocity() / 2.0)


print(f"robot_parts length: {len(robot_parts)}") 
base_elements=["base_link", "base_link_Torso_joint", "Torso", "torso_lift_joint", "torso_lift_link", "torso_lift_link_TIAGo front arm_11367_joint", "TIAGo front arm_11367"]
my_chain = Chain.from_urdf_file("tiago_urdf.urdf", base_elements=["base_link", "base_link_Torso_joint", "Torso", "torso_lift_joint", "torso_lift_link", "torso_lift_link_TIAGo front arm_11367_joint", "TIAGo front arm_11367"])

for link_id in range(len(my_chain.links)):

    # This is the actual link object
    link = my_chain.links[link_id]
    
    # I've disabled "torso_lift_joint" manually as it can cause
    # the TIAGO to become unstable.
    if link.name not in part_names or  link.name =="torso_lift_joint":
        print("Disabling {}".format(link.name))
        my_chain.active_links_mask[link_id] = False
        
# Initialize the arm motors and encoders.
motors = []
for link in my_chain.links:
    if link.name in part_names and link.name != "torso_lift_joint":
        motor = robot.getDevice(link.name)

        # Make sure to account for any motors that
        # require a different maximum velocity!
        if link.name == "torso_lift_joint":
            motor.setVelocity(0.07)
        else:
            motor.setVelocity(1)
            
        position_sensor = motor.getPositionSensor()
        position_sensor.enable(timestep)
        motors.append(motor)

# The Tiago robot has a couple more sensors than the e-Puck
# Some of them are mentioned below. We will use its LiDAR for Lab 5

rng = robot.getDevice('range-finder')
rng.enable(timestep)
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
# mode = 'autonomous'
mode = 'picknplace'

target_item_list = ["orange"]
vrb = True

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
    world_x = -12 + (map_x / 30)
    world_y = 0 - (map_y / 30)
    
    return world_x, world_y

def world_to_map(world_x, world_y):
    map_x = int((world_x + 12) * 30)
    map_y = int(-world_y * 30)
    
    return map_x, map_y

def flip_xy(orig_x, orig_y):
    return (orig_y, orig_x)

def set_arm_to_default_pose():
   
    print("arm to default pose")

    # Define the default joint positions
    default_positions = {
        "arm_1_joint": 0.0700,
        "arm_2_joint": 1.0200,
        "arm_3_joint": -3.1600,
        "arm_4_joint": 1.2700,
        "arm_5_joint": 1.3200,
        "arm_6_joint": 0.0000,
        "arm_7_joint": 1.4100
    }


    for joint_name, position in default_positions.items():
        motor = robot.getDevice(joint_name)
        if motor:
            motor.setPosition(position)
            print(f"Setting {joint_name} to {position:.4f} rad ({math.degrees(position):.1f}°)")

    positioning_timer = 0
    while robot.step(timestep) != -1 and positioning_timer < 100:
        positioning_timer += 1

        # robot_parts[MOTOR_LEFT].setVelocity(0)
        # robot_parts[MOTOR_RIGHT].setVelocity(0)

    print("arm reset complete")
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
    kernel = np.ones((22, 22)) 
    map_cspace = convolve2d(filtered_map, kernel, mode='same', boundary='wrap')
    map_cspace = map_cspace > 0
    # start = (316,300)
    # start = (200, 200)
    start = world_to_map(-8.736592, -4.648618)
    # start = world_to_map(gps.getValues()[0], gps.getValues()[1])
    # end = (100,123)
    end = (204, 223)
    # end = (300, 50) #flipped from map visualization

    path = dijkstra(map_cspace, start, end)
    waypoints = [map_to_world(x, y) for (x, y) in path]

    np.save("../../maps/path.npy", waypoints)
    map_display = map_cspace.copy()
    map_display[10, 20] = 5
    if path:
        print("Path saved!")
    else:
        print("No valid path found")
    for (x, y) in path:
        map_display[x, y] = 0.5
    
    plt.figure(figsize=(6,6))
    plt.title("Path Visualization")
    plt.imshow(map_display, cmap='gray')
    
    plt.show()

state = 0 # use this to iterate through your path
elapsed_time = 0
forward_state = 0

def lookForTarget(target_item, recognized_objects):
    if len(recognized_objects) > 0:

        for item in recognized_objects:
            if target_item in str(item.getModel()):

                target = recognized_objects[0].getPosition()
                dist = abs(target[2])

                if dist < 5:
                    return True

def checkArmAtPosition(ikResults, cutoff=0.01):
    '''Checks if arm at position, given ikResults'''
    
    # Get the initial position of the motors
    initial_position = [0,0,0,0] + [m.getPositionSensor().getValue() for m in motors] + [0,0,0,0]

    # Calculate the arm
    arm_error = 0
    for item in range(14):
        arm_error += (initial_position[item] - ikResults[item])**2
    arm_error = math.sqrt(arm_error)
    print(f'ARM ERRRROR: {arm_error} -----------------------')
    print(f'BOOOOOL: {arm_error < 0.07}-------------------------')
    if arm_error < 0.07:
        if vrb:
            print("Arm at position.")
        print("RETURNINNNNNG TRUE HELP")
        return True
    return False


def moveArmToTarget(ikResults):
    '''Moves arm given ikResults'''
    # Set the robot motors
    print("IK RESULTS TYPE:", type(ikResults))  # Debugging
    print("IK RESULTS VALUE:", ikResults)
    for res in range(len(ikResults)):
        if my_chain.links[res].name in part_names:
            # This code was used to wait for the trunk, but now unnecessary.
            # if abs(initial_position[2]-ikResults[2]) < 0.1 or res == 2:
            current_pos = robot.getDevice(my_chain.links[res].name).getPositionSensor().getValue()
            desired_pos = ikResults[res]
            print(f"Joint {my_chain.links[res].name}: Desired = {desired_pos:.4f} rad, Current = {current_pos:.4f} rad")
            robot.getDevice(my_chain.links[res].name).setPosition(ikResults[res])
            if vrb:
                print("Setting {} to {}".format(my_chain.links[res].name, ikResults[res]))

def calculateIk(offset_target,  orient=True, orientation_mode="Y", target_orientation=[0,0,1]):
    '''
    This will calculate the iK given a target in robot coords
    Parameters
    ----------
    param offset_target: a vector specifying the target position of the end effector
    param orient: whether or not to orient, default True
    param orientation_mode: either "X", "Y", or "Z", default "Y"
    param target_orientation: the target orientation vector, default [0,0,1]

    Returns
    ----------
    rtype: bool
        returns: whether or not the arm is at the target
    '''

    # Get the initial position of the motors
    initial_position = [0,0,0,0] + [m.getPositionSensor().getValue() for m in motors] + [0,0,0]
    
    # Calculate IK
    ikResults = my_chain.inverse_kinematics(offset_target, initial_position=initial_position,  target_orientation = [0,0,1], orientation_mode="Y")
    print("Desired IK positions:")
    for i, joint in enumerate(my_chain.links):
        print(f"{joint.name}: Desired = {ikResults[i]:.4f} rad")
    # Use FK to calculate squared_distance error
    position = my_chain.forward_kinematics(ikResults)

    # This is not currently used other than as a debug measure...
    squared_distance = math.sqrt((position[0, 3] - offset_target[0])**2 + (position[1, 3] - offset_target[1])**2 + (position[2, 3] - offset_target[2])**2)
    print("IK calculated with error - {}".format(squared_distance))

    # Reset the ikTarget (deprec)
    # ikTarget = offset_target
    
    return ikResults
        
def getTargetFromObject(recognized_objects):
    ''' Gets a target vector from a list of recognized objects '''

    # Get the first valid target
    target = recognized_objects[0].getPosition()

    # Convert camera coordinates to IK/Robot coordinates
    offset_target = [-(target[2])+0.22, -target[0]+0.06, (target[1])+0.97+0.2]

    return offset_target

def reachArm(target, previous_target, ikResults, cutoff=0.00005):
    '''
    This code is used to reach the arm over an object and pick it up.
    '''

    # Calculate the error using the ikTarget
    error = 0
    ikTargetCopy = previous_target

    # Make sure ikTarget is defined
    if previous_target is None:
        error = 100
    else:
        for item in range(3):
            error += (target[item] - previous_target[item])**2
        error = math.sqrt(error)

    
    # If error greater than margin
    if error > 0.05:
        print("Recalculating IK, error too high {}...".format(error))

        ikResults = calculateIk(target)
        ikTargetCopy = target
        moveArmToTarget(ikResults) 

    # Exit Condition
    if checkArmAtPosition(ikResults):
        
        if vrb:
            print("NOW SWIPING")
        return [True, ikTargetCopy, ikResults]
    else:
        if vrb:
            print("ARM NOT AT POSITION")

    # Return ikResults
    return [False, ikTargetCopy, ikResults]

def closeGrip():
    robot.getDevice("gripper_right_finger_joint").setPosition(0.0)
    robot.getDevice("gripper_left_finger_joint").setPosition(0.0)

def openGrip():
    robot.getDevice("gripper_right_finger_joint").setPosition(0.045)
    robot.getDevice("gripper_left_finger_joint").setPosition(0.045)

def rotate_y(x,y,z,theta):
    new_x = x*np.cos(theta) + y*np.sin(theta)
    new_z = z
    new_y = y*-np.sin(theta) + x*np.cos(theta)
    return [-new_x, new_y, new_z]

def world_to_robot(x,y,z, robot_pose_x, robot_pose_y, world_theta):
    # robot_pose_x = gps.getValues()[0]
    # robot_pose_y = gps.getValues()[1]

    object_with_offset_y = y + 0.15 # 0.05 is radius of the orange
    object_with_offset_x = x - 0.05

    robot_pose_with_offset_x = robot_pose_x + 0.2
    robot_pose_with_offset_y = robot_pose_y + 0.2

    print('world to robot x-coord ', robot_pose_x)
    print('world to robot y-coord ', robot_pose_y)

    new_x = (object_with_offset_x - robot_pose_with_offset_x) * math.cos(world_theta) + (object_with_offset_y - robot_pose_with_offset_y) * math.sin(world_theta)
    new_y = (object_with_offset_x - robot_pose_with_offset_x) * -math.sin(world_theta) + (object_with_offset_y - robot_pose_with_offset_y)* math.cos(world_theta)
    
    return [new_x , new_y , z-0.05]


if mode == 'picknplace':
    # Part 4: Use the function calls from lab5_joints using the comments provided there
    ## use path_planning to generate paths
    ## do not change start_ws and end_ws below
    start_ws = [(3.7, 5.7)]
    end_ws = [(10.0, 9.3)]

    waypoints = [] # Replace with code to load your path
    filtered_map = np.load("../../maps/mapv1.npy")
    kernel = np.ones((22, 22)) 
    map_cspace = convolve2d(filtered_map, kernel, mode='same', boundary='wrap')
    map_cspace = map_cspace > 0
    # start = (204, 223)
    start = world_to_map(-5.389170, -7.199493) 


    # end = world_to_map(-8.736592, -4.648618) // this is starting position of robot??
    # end = world_to_map(-8.47522, -4.9)
    # start = (116, 180)
    # end = (116, 180)
    end = world_to_map(-7.53053, -5.0713)
    # end = (128, 163)
    # end = (300, 50) #flipped from map visualization

    path = dijkstra(map_cspace, start, end)
    waypoints = [map_to_world(x, y) for (x, y) in path]

    np.save("../../maps/path.npy", waypoints)
    map_display = map_cspace.copy()
    map_display[10, 20] = 5
    if path:
        print("Path saved!")
    else:
        print("No valid path found")
    for (x, y) in path:
        map_display[x, y] = 0.5
    
    plt.figure(figsize=(6,6))
    plt.title("Path Visualization")
    plt.imshow(map_display, cmap='gray')
    
    plt.show()
    
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
    world_theta = pose_theta + math.pi/2

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
            err_margin = .5
            turn_speed = .07 # default value will change this
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
        
        def reach_position(distance_to_goal, is_proportional=True) -> tuple:
            """goes foward until the distance is within the error -> ruturns true is it has reached the goal"""
            # global leftMotor, rightMotor, leftMax, rightMax
            
            # tuning variables r here
            err_margin = .5
            foward_speed = .1
            min_foward_speed = .01
            max_foward_speed = .2
            
            if is_proportional:
                portional_gain = 5
                foward_speed = abs(distance_to_goal * portional_gain)
            
            if foward_speed > max_foward_speed:
                foward_speed = max_foward_speed
                
            if foward_speed < min_foward_speed:
                foward_speed = min_foward_speed
                
            if not (distance_to_goal < err_margin and distance_to_goal > -err_margin):
                return (MAX_SPEED * foward_speed, MAX_SPEED * foward_speed)

            return None

        # get the times:
        elapsed_time += timestep / 1000.0
        delta_time = timestep / 1000.0
        
        # TODO: controller / calculations
        
        # euclidian distance
        # goal_pos = (-0.19, 0.125162, 0)
        # filtered_waypoints = waypoints[3::4]
        if len(waypoints) == 0:
            vL = 0
            vR = 0
            continue

        goal_pos = waypoints[state]
        # goal_pos = filtered_waypoints[state]
        # print('WAYPOINTS', waypoints)
        print('velocities', vL, vR)
        # print('FILTERED WAYPOINTS', filtered_waypoints)
        euc_dis = math.pow(goal_pos[0] - pose_x, 2)
        euc_dis += math.pow(goal_pos[1] - pose_y, 2)
        euc_dis = math.sqrt(euc_dis)
        
        # calculate the angle to goal
        ang_to_goal = math.atan2(goal_pos[1] - pose_y, goal_pos[0] - pose_x)
        print('ANGLEASD ASF ', ang_to_goal)
        print('GOALS', goal_pos)
        print('WAYPOINTS LENGTH', len(waypoints))
        ang_to_goal = (ang_to_goal - world_theta + math.pi) % (2 * math.pi) - math.pi
 
        vels = turn_to_goal(ang_to_goal, True)
        if vels is None:
            vels = reach_position(euc_dis, True)
            if vels is None:
                if mode == 'picknplace':
                    if objectGrabbed == True:
                        state -= 25
                        objectGrabbed = False
                        retracing = True
                    elif retracing == True:
                        state -= 1
                    elif state < len(waypoints)-1:
                        state += 1
                    forward_state += 1
                    vels = (0, 0) 


        if objectGrabbed:
            target_pos = (-0.2, 0.0, 0.09, 0.07, 1.02, -3.16, 1.27, 1.32, 0.0, 1.41, 'inf', 'inf')
            
            robot_parts=[]
            print("RESETTING ARM JOINTS")
            
            # for i in range(10):
            #     robot.step(timestep)
            closeGrip()
            

            for i in range(10):
                robot.step(timestep)
                
            set_arm_to_default_pose()
            # for i in range(N_PARTS):
                # robot_parts.append(robot.getDevice(part_names[i]))
                # robot_parts[i].setPosition(float(target_pos[i]))
                # robot_parts[i].setVelocity(robot_parts[i].getMaxVelocity() / 2.0)


        if mode == 'picknplace':
            # if objectGrabbed == True:
            #     if state > 0:
            #         state -= 1

            if state == len(waypoints)-1 and objectGrabbed == False:
                vels = (0, 0)
                objects = camera.getRecognitionObjects()
                if objects:
                    print(f"Recognized {len(objects)} objects:")
                    for obj in objects:
                        print(f" - Model: {obj.getModel()}")
                        print(f" - Position: {obj.getPosition()}")
                        print(f" - Size: {obj.getSize()}")
                        print(f" - Rotation: {obj.getOrientation()}")
                        print("---------------------------")
                else:
                    print("No objects detected.")

                # print('RECOGNIZED OBJECT: ', recognized_objects[0].getPosition())
                if lookForTarget('orange', objects):
                    objectSpotted = True
                    vels = (0, 0)

                if objectSpotted == True:
                    coords = world_to_robot(-7.50701, -6.04787, 0.889765, pose_x, pose_y, world_theta)
                    print('OBJECT COORDS: ', coords)
                    # arm_target = getTargetFromObject(coords)
                    # coords = rotate_y(*coords, np.radians(90))
                    if ikCalculated == False:
                        ikResults = calculateIk(coords)
                        print('IK RESULTS: ', ikResults)
                        ikCalculated = True
                    
                    if ikResults is not None:
                    # if moveArmRan == False:
                        openGrip() 
                        print("Opening")
                        reach_arm_results = reachArm(coords, None, ikResults, cutoff=0.0005)
                        print(f'reach arm result: {reach_arm_results[0]}')
                        # reachArmRan = True
                        # moveArmToTarget(ikResults)
                        # moveArmRan = True

                        # else:
                        if reach_arm_results[0] and objectReached == False:
                            objectReached = True
                            grip_close_start_time = robot.getTime()
                            print('arm reached target and now closing gripper')
                            closeGrip()
                            for i in range(30):
                                robot.step(timestep)
                            
                        if objectReached and robot.getTime() - grip_close_start_time >= 3:
                            objectGrabbed = True
                            print('gripper closed')
                        else:
                            print('failed to reach target')
                    else:
                        print('IK calc failed')

                    # arm_target = getTargetFromObject(objects)

                    # ikResults = calculateIk(arm_target)
                    
                    # # arm_rotated_for_world = rotate_y(*arm_target, -math.pi/2)
                    # # ikResults = calculateIk(arm_rotated_for_world)

                    # if ikResults is not None:
                    #     # turn_to_goal(math.pi/2)
                    #     ang_to_goal = 90
                    #     reach_arm_results = reachArm(arm_target, None, ikResults, cutoff=0.00005)
                    #     # moveArmToTarget(ikResults)

                    #     # if reach_arm_results[0]:
                    #     #     print('arm reached target and now closing gripper')
                    #     #     closeGrip()
                    #     #     print('gripper closed')
                    #     # else:
                    #     #     print('failed to reach target')
                    # else:
                    #     print('IK calc failed')
                else:
                    vels = (0.15 * -MAX_SPEED, 0.15 * MAX_SPEED)   

        vL = vels[0]
        vR = vels[1]
        
        
        
        
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
