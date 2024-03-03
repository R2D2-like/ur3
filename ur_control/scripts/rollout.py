#!/usr/bin/env python3
from ur_control.arm import Arm
import rospy
import numpy as np
import os

class Rollout:
    def __init__(self) -> None:
        rospy.init_node('rollout', anonymous=True)
        self.arm = Arm(gripper_type=None, ee_link='wrist_3_link') # with gripper
        self.data = None
        self.save_dir = '/root/Research_Internship_at_GVlab/real_exp/rollout/'
        self.save_name = input('Enter the name of the save file: ')
        rospy.loginfo('Rollout node initialized')

    def rollout(self):
        ee_position = np.load(self.save_dir + self.save_name) #(2000, 3)
        target_time = 0.01
        for i in range(ee_position.shape[0]):
            pose_goal = self.arm.end_effector()
            pose_goal[0] = ee_position[i, 0]
            pose_goal[1] = ee_position[i, 1]
            pose_goal[2] = ee_position[i, 2]
            try:
                self.arm.set_target_pose(pose=pose_goal, wait=True, target_time=target_time)
            except Exception as e:
                print(e)

        rospy.loginfo('Rollout completed')

if __name__ == '__main__':
    rollout = Rollout()
    rollout.rollout()
        