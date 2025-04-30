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
        self.robot = robot_reference
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

    def move_arm_to_position(self, target=[0, -.5, 0], plot = False):
        """center of arm part 0, -.5, .5
        robot -y, z, -x"""
        offset_target = [-(target[2])+0.22, -target[0]+0.08, (target[1])+0.97+0.2]
        initial_position = [0,0,0,0] + [m.getPositionSensor().getValue() for m in self.motors] + [0,0,0,0]
        ikResults = self.armChain.inverse_kinematics(offset_target, initial_position=initial_position,  
                                                    target_orientation = [0,0,1], orientation_mode="Y")
        for res in range(len(ikResults)):
            if self.armChain.links[res].name in self.part_names:
                self.robot.getDevice(self.armChain.links[res].name).setPosition(ikResults[res])
                print("Setting {} to {}".format(self.armChain.links[res].name, ikResults[res]))
        if plot:
            self.plot_3d(initial_position, offset_target, ikResults)
                
    def drop_in_basket(self):
        """moves the arm to the basket position, make sure to add timer delay
        doesn't open the claw either, do that at the end"""
        target = [0, -.7, -.05]
        self.move_arm_to_position(target)

    def closeGrip(self):
        self.robot.getDevice("gripper_right_finger_joint").setPosition(0.0)
        self.robot.getDevice("gripper_left_finger_joint").setPosition(0.0)

    def openGrip(self):
        self.robot.getDevice("gripper_right_finger_joint").setPosition(0.045 )
        self.robot.getDevice("gripper_left_finger_joint").setPosition(0.045)
                

    def plot_3d(self, initial_position, offset_target, ikResults):
        ax = matplotlib.pyplot.figure().add_subplot(111, projection='3d')
        self.armChain.plot(initial_position, ax, target=offset_target)
        self.armChain.plot(ikResults, ax, target=offset_target)
        matplotlib.pyplot.show()