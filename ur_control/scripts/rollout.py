#!/usr/bin/env python3
from ur_control.arm import Arm
import rospy
import numpy as np
import os
import sys
# add /root/Research_Internship_at_GVlab/scripts/config
sys.path.append('/root/Research_Internship_at_GVlab/scripts/config')
from config import SCALING_FACTOR, DEMO_TRAJECTORY_MIN, DEMO_TRAJECTORY_MAX
import torch

class Rollout:
    def __init__(self) -> None:
        rospy.init_node('rollout', anonymous=True)
        self.arm = Arm(gripper_type=None, ee_link='wrist_3_link') # with gripper
        self.data = None
        self.save_dir = '/root/Research_Internship_at_GVlab/real/'
        self.exploratory_data_name = input('Enter the name of the exploratory data file: ')
        self.save_name = input('Enter the name of the save file for inference results: ')
        rospy.loginfo('Rollout node initialized')

    def output2position(self, normalized_output):
        # 正規化された出力をもとの値に戻す
        # normalized_output = np.load(self.save_dir + self.save_name) #(2000, 3)
        normalized_output /= SCALING_FACTOR
        output = normalized_output * (DEMO_TRAJECTORY_MAX - DEMO_TRAJECTORY_MIN) + DEMO_TRAJECTORY_MIN
        return output
    
    def infer_eef_position(self):
        # load data
        data = np.load(self.save_dir + 'rollout/data/exploratory/'+ self.exploratory_data_name + '.npy') # normalized
        # load model
        model_weights_path = self.save_dir + 'model/baseline.pth'
        model = torch.load(model_weights_path)
        model.eval()
        # inference
        output = model(torch.tensor(data).float())
        output = output.detach().numpy()
        eef_position = self.output2position(output)
        np.save(self.save_dir + 'rollout/data/inferred/eef_position_'+ self.save_name + '.npy', eef_position)
        rospy.loginfo('Inference completed')
        return eef_position #(2000, 3)

    def rollout(self):
        ee_position = self.infer_eef_position() #(2000, 3)
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
        