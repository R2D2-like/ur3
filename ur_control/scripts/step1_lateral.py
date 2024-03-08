#!/usr/bin/env python3
from ur_control.arm import Arm
import rospy
import numpy as np
import os
from ur_control import transformations, traj_utils, conversions


class Step1Lateral:
    def __init__(self):
        rospy.init_node('/step1/lateral', anonymous=True)
        self.arm = Arm(gripper_type=None, ee_link='wrist_3_link') # with gripper
        self.arm.set_ft_filtering()
        self.arm.zero_ft_sensor()
        self.lateral_movements_data = None
        mode = input('0:collecting data, 1:rollout: ')
        if mode == '0':
            self.save_dir = '/root/Research_Internship_at_GVlab/real/step1/data/lateral/'
        else:
            self.save_dir = '/root/Research_Internship_at_GVlab/real/rollout/data/exploratory/lateral/'
        self.save_name = input('Enter the name of the save file: ')
        rospy.loginfo('Step1 lateral node initialized')

    def move_endeffector(self, deltax, target_time):
        # get current position of the end effector
        cpose = self.arm.end_effector()
        # define the desired translation/rotation
        deltax = np.array(deltax)
        # add translation/rotation to current position
        cpose = transformations.transform_pose(cpose, deltax, rotated_frame=True)
        # execute desired new pose
        # may fail if IK solution is not found
        self.arm.set_target_pose(pose=cpose, wait=True, target_time=target_time)

    def go_to_initial_pose(self):
        joint_positions = [1.57, -1.57, 1.57, -1.57, -1.57, 0]
        self.arm.set_joint_positions(positions=joint_positions, wait=True, target_time=0.5)

    def lateral_movements(self):
        self.arm.zero_ft_sensor()
        self.move_endeffector([0.05, 0, 0, 0, 0, 0], target_time=1)
        self.lateral_movements_data = self.arm.get_wrench_history(hist_size=100)
        rospy.sleep(1)
        self.move_endeffector([-0.05, 0, 0, 0, 0, 0], target_time=1)
        self.lateral_movements_data = np.concatenate((self.lateral_movements_data, self.arm.get_wrench_history(hist_size=100))) 

    def save_data(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        np.save(self.save_dir + self.save_name + '.npy', self.lateral_movements_data) # (200, 6)
        rospy.loginfo('Data saved')


if __name__ == '__main__':
    is_sim = input('Is this a simulation? (y/n): ')
    if is_sim == 'y':
        rospy.loginfo('This is a simulation')
        is_sim = True
    else:
        rospy.loginfo('This is a real robot')
        is_sim = False
    step1_lateral = Step1Lateral()
    if is_sim:
        step1_lateral.go_to_initial_pose()
    rospy.loginfo('start step1 lateral')
    step1_lateral.lateral_movements()
    step1_lateral.save_data()
    rospy.loginfo('Step 1 lateral completed')

        