import math

MAX_SPEED = 7.0  # [rad/s]
MAX_SPEED_MS = 0.633 # [m/s]
AXLE_LENGTH = 0.4044 # m
MOTOR_LEFT = 10
MOTOR_RIGHT = 11
N_PARTS = 12
LIDAR_ANGLE_BINS = 667
LIDAR_SENSOR_MAX_RANGE = 5.5 # Meters
LIDAR_ANGLE_RANGE = math.radians(240)

# not currently needed for our implementation
def inverse_wheel_kinematics(distance, delta_theta, timestep, axle_diameter=AXLE_LENGTH):
    """takes in the amount to travel then gives the rotations we need"""
    delta_time = timestep / 1000.0
    if delta_time == 0: return 0, 0
    
    # reversed the equations we had from last lab 2 for odometry
    v_linear = distance / delta_time  
    vR = (delta_theta * axle_diameter) / (2 * delta_time) + v_linear
    vL = v_linear - (delta_theta * axle_diameter) / (2 * delta_time)
    return vL, vR

def turn_to_goal(ang_to_goal: float, is_proportional=True) -> tuple:
    """ takes in the angle and turns if not facing -> returns true if it is facing the goal otherwise returns false"""
    
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


def nav_to_waypoint(waypoints, curr_waypoint, pose_x, pose_y, pose_theta):
    if len(waypoints) == 0:
        print('waypoints is currently none')
        return (0, 0)
    
    print('curr waypoint:', waypoints[curr_waypoint])

    goal_pos = waypoints[curr_waypoint]
    euc_dis = math.pow(goal_pos[0] - pose_x, 2)
    euc_dis += math.pow(goal_pos[1] - pose_y, 2)
    euc_dis = math.sqrt(euc_dis)

    # calculate the angle to goal
    ang_to_goal = math.atan2(goal_pos[1] - pose_y, goal_pos[0] - pose_x)
    print('ANGLE', ang_to_goal)
    print('GOALS', goal_pos)
    print('WAYPOINTS LENGTH', len(waypoints))
    
    ang_to_goal = (ang_to_goal - pose_theta + math.pi) % (2 * math.pi) - math.pi

    vels = turn_to_goal(ang_to_goal, True)
    if vels is None:
        vels = reach_position(euc_dis, True)
        if vels is None:
            if curr_waypoint < len(waypoints)-1:
                curr_waypoint += 1
            vels = (0, 0)

    return vels