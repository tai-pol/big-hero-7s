import sys
import tempfile
try:
    import ikpy
    from ikpy.chain import Chain
except ImportError:
    sys.exit('The "ikpy" Python module is not installed. '
             'To run this sample, please upgrade "pip" and install ikpy with this command: "pip install ikpy"')

import math
from controller import robot, Robot, Keyboard # type:ignore

if ikpy.__version__[0] < '3':
    sys.exit('The "ikpy" Python module version is too old. '
             'Please upgrade "ikpy" Python module to version "3.0" or newer with this command: "pip install --upgrade ikpy"')
import matplotlib.pyplot


class Gripper:
    def __init__(self, robot_reference):
        """
        Initializes the Gripper class with a reference to the robot.
        
        :param robot_reference: A reference to the robot object.
        """
        self.robot = robot_reference
        self.target = [0, -.7, -.05]
        self.timestep = int(self.robot.getBasicTimeStep())
        
        self.base_elements=["base_link", "base_link_Torso_joint", "Torso", "torso_lift_joint",
                            "torso_lift_link", "torso_lift_link_TIAGo front arm_joint", "TIAGo front arm"]
        self.armChain = Chain.from_urdf_file("tiago_urdf.urdf", last_link_vector=[0.004, 0,-0.1741], base_elements=self.base_elements)
        self.part_names = ("head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
                        "arm_2_joint",  "arm_3_joint",  "arm_4_joint", "arm_5_joint",
                        "arm_6_joint",  "arm_7_joint",  "wheel_left_joint", "wheel_right_joint")
        self.motors = []
        for link in self.armChain.links:
            if link.name in self.part_names and link.name != "torso_lift_joint":
                motor = self.robot.getDevice(link.name)
                motor.setVelocity(1)
                position_sensor = motor.getPositionSensor()
                position_sensor.enable(self.timestep)
                self.motors.append(motor)
        
        # self.move_arm_to_position([0, -.7,  -.25])

            
    def move_arm_to_position(self, target=[0, -.5, 0], plot = False):
        """
        Moves the arm to a specified target position.
        
        :param target: The target position to move the arm to. Defaults to [0, -.5, 0].
        :param plot: If True, plots the arm movement. Defaults to False.
        """
        
        """center of arm part 0, -.5, .5
        robot -y, z, -x"""
        
        # print(self.armChain.forward_kinematics([0,0,0,0] + [m.getPositionSensor().getValue() for m in self.motors] + [0,0,0]))
        self.target = target
        try:
            offset_target = [-(target[2])+0.22, -target[0]+0.08, (target[1])+0.97+0.2]
            initial_position = [0,0,0,0] + [m.getPositionSensor().getValue() for m in self.motors] + [0,0,0,0]
            ikResults = self.armChain.inverse_kinematics(offset_target, initial_position=initial_position,  
                                                        target_orientation = [0,0,1], orientation_mode="Y")
            
            for res in range(len(ikResults)):
                if self.armChain.links[res].name in self.part_names:
                    self.robot.getDevice(self.armChain.links[res].name).setPosition(ikResults[res])
                    # print("Setting {} to {}".format(self.armChain.links[res].name, ikResults[res]))
            if plot:
                self.plot_3d(initial_position, offset_target, ikResults)
        except Exception as e:
            print(e)
                
    def move_to_basket(self):
        """
        Moves the arm to the basket position.
        """
        self.target = [0, -.7, -.05]
        self.move_arm_to_position(self.target)

    def closeGrip(self):
        """
        Closes the gripper.
        """
        self.robot.getDevice("gripper_right_finger_joint").setPosition(0.0)
        self.robot.getDevice("gripper_left_finger_joint").setPosition(0.0)

    def openGrip(self):
        """
        Opens the gripper.
        """
        self.robot.getDevice("gripper_right_finger_joint").setPosition(0.045 )
        self.robot.getDevice("gripper_left_finger_joint").setPosition(0.045)

    def plot_3d(self, initial_position, offset_target, ikResults):
        """
        Plots the arm movement in 3D.
        
        :param initial_position: The initial position of the arm.
        :param offset_target: The offset target position.
        :param ikResults: The results of the inverse kinematics calculation.
        """
        ax = matplotlib.pyplot.figure().add_subplot(111, projection='3d')
        self.armChain.plot(initial_position, ax, target=offset_target)
        self.armChain.plot(ikResults, ax, target=offset_target)
        matplotlib.pyplot.show()
    
    def tele_increment(self, amountxyz: list[int]):
        """
        Increments the arm and moves it by a specified amount in the x, y, z directions.
        
        :param amountxyz: A list of three integers specifying the amount to move in the x, y, z directions.
        """
        # y, z, -x = amount 
        
        # can't hit anything otherwise it fudging breaks
        # needs to be 1m from the shelf to grab correctly
        
        tuning = [.005, .005, .005]
        amountxyz = [amountxyz[1] * tuning[0], amountxyz[2] * tuning[1], -amountxyz[0] * tuning[2]]
        
        self.target[0] += amountxyz[0]
        self.target[1] += amountxyz[1]
        self.target[2] += amountxyz[2]
        
        # print("moving to: ", self.target)
        self.move_arm_to_position(self.target)
    
    def reset_motors(self):
        """
        Resets the motors to their initial positions.
        """
        part_names = ("head_2_joint", "head_1_joint", "torso_lift_joint", "arm_1_joint",
              "arm_2_joint",  "arm_3_joint",  "arm_4_joint",      "arm_5_joint",
              "arm_6_joint",  "arm_7_joint",  "wheel_left_joint", "wheel_right_joint",
              "gripper_left_finger_joint","gripper_right_finger_joint")
        
        target_pos = (0.0, 0.0, 0, 0.07, 1.02, 0, 1.27, 1.32, 0.0, 1.41, 'inf', 'inf',0.045,0.045)

        robot_parts={}
        for i, part_name in enumerate(part_names):
            robot_parts[part_name]= self.robot.getDevice(part_name)
            robot_parts[part_name].setPosition(float(target_pos[i]))
            robot_parts[part_name].setVelocity(robot_parts[part_name].getMaxVelocity() / 2.0)