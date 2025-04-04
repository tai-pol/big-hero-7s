from controller import Robot, DistanceSensor, Motor, LightSensor

# time in [ms] of a simulation step
TIME_STEP = 64

MAX_SPEED = 6.28

# create the Robot instance.
robot = Robot()

# initialize distance sensors
ps = []
psNames = [
    'ps0', 'ps1', 'ps2', 'ps3',
    'ps4', 'ps5', 'ps6', 'ps7'
]

ls = []
lsNames = [
    'ls0', 'ls1', 'ls2', 'ls3',
    'ls4', 'ls5', 'ls6', 'ls7'
]

for i in range(8):
    ps.append(robot.getDevice(psNames[i]))
    ps[i].enable(TIME_STEP)
    
    ls.append(robot.getDevice(lsNames[i]))
    ls[i].enable(TIME_STEP)

# initialize motors
left_motor = robot.getDevice('left wheel motor')
right_motor = robot.getDevice('right wheel motor')
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

FOLLOW_WALL_LEFT = 0
TURN_RIGHT = 1
TURN_LEFT = 2
FOLLOW_WALL_RIGHT = 3
TURN_AROUND = 4
STOP = 5

RIGHT_TURN = 6

state = FOLLOW_WALL_LEFT
# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(TIME_STEP) != -1:
    
    ps_values = []
    ls_values = []
    for i in range(8):
        ps_values.append(ps[i].getValue())
        ls_values.append(ps[i].getValue())
    #print("light sensor: ",ls_values)
    print("distance sensors: ", ps_values)
    # top_left = 
    
    #used to keep left while turning
    stay_left = ps_values[5] >100.0 #or ps_values[2] <80
    
    #used for epuck to advance forward
    left_obstacle =  ps_values[5] > 80.0 or ps_values[6] >80.0 #or ps_values[7] > 70.0
    right_obstacle = ps_values[2] > 80.0 or ps_values[1] > 80.0 #ps_values[1] > 80.0 or 
    front_obstacle = ps_values[7] > 80.0 or ps_values[0] > 80.0 #or ps_values[6] >60.0
   
    left_speed = 0.5 * MAX_SPEED
    right_speed = 0.5 * MAX_SPEED
    if state == FOLLOW_WALL_LEFT: 
        if right_obstacle:
            # obstacle in front, turn right
            state = TURN_LEFT
        # elif front_obstacle and left_obstacle:
           # state = RIGHT_TURN 
       
        elif front_obstacle:
            state = TURN_RIGHT
        elif stay_left:
             state = TURN_LEFT
            
    elif state == TURN_RIGHT:
         
         left_speed = 0.5 * MAX_SPEED
         right_speed = -0.5 * MAX_SPEED
         
         if left_obstacle:
             state = FOLLOW_WALL_LEFT
    elif state == TURN_LEFT:
         left_speed = -.4 *MAX_SPEED
         right_speed = 0 *MAX_SPEED
         
         if left_obstacle:
             state = FOLLOW_WALL_LEFT
             
    # elif state == RIGHT_TURN:
         # left_speed = 0
         # right_speed = 0
         # radians = (200 / 360) 
         # right_motor.setPosition(radians)
         # #_motor.setPosition(-radians)
         
         # if stay_left:
             # state = FOLLOW_WALL_LEFT
       
     # elif state == T
    print("state: ", state)
    print("front_ob bool:", front_obstacle)
    print("left_ob bool:", left_obstacle)
    print("right_ob bool:", right_obstacle)    
    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)
        
        
        
        
   

# Enter here exit cleanup code.
