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
from geometry_msgs.msg import Pose

class RolloutBaseline:
    def __init__(self) -> None:
        rospy.init_node('rollout_baseline', anonymous=True)
        self.arm = Arm(gripper_type=None, ee_link='wrist_3_link') # with gripper

        # Subscriber
        self.sub_eef_pose = rospy.Subscriber('/eef_pose', Pose, self.eef_pose_callback)
        self.eef_pose_history = []

        self.base_dir = '/root/Research_Internship_at_GVlab/real/'
        stiffness = input('stiffness level (1, 2, 3, 4): ')
        friction = input('friction level (1, 2, 3): ')
        self.sponge = 's' + stiffness + 'f' + friction
        self.base_save_dir = self.base_dir + 'rollout/data/'
        rospy.loginfo('Rollout node initialized')

    def eef_pose_callback(self, msg):
        current_pos = np.array([msg.position.x, msg.position.y, msg.position.z, \
                                msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        self.eef_pose_history.append(current_pos)

    def output2position(self, normalized_output):
        # 正規化された出力をもとの値に戻す
        # normalized_output = np.load(self.base_dir + self.save_name) #(2000, 3)
        normalized_output /= SCALING_FACTOR
        output = normalized_output * (DEMO_TRAJECTORY_MAX - DEMO_TRAJECTORY_MIN) + DEMO_TRAJECTORY_MIN
        return output
    
    def predict_eef_position(self):
        # load data
        data = np.load(self.base_dir + 'rollout/data/exploratory/exploratory_action_preprocessed.npz')[self.sponge] # normalized
        # load model
        model_weights_path = self.base_dir + 'model/baseline/baseline_model.pth'
        model = torch.load(model_weights_path)
        model.eval()
        # inference
        output = model(torch.tensor(data))
        output = output.detach().numpy()
        eef_position = self.output2position(output)
        save_dir = self.base_save_dir + 'predicted/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.savez(save_dir + self.sponge + '.npz', eef_position=eef_position)
        print('Data saved at\n: ', save_dir + self.sponge + '.npz')
        rospy.loginfo('Inference completed')
        return eef_position #(2000, 3)

    def rollout(self):
        ee_position = self.predict_eef_position() #(2000, 3)
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

        traj_history = self.eef_pose_history[-2000:] # (2000, 7)
        ft_history = self.arm.get_wrench_history(hist_size=2000) # (2000, 6)
        save_dir = self.base_save_dir + 'result/' 
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = save_dir + self.sponge + '.npz'
        np.savez(save_path, pose=traj_history, ft=ft_history)
        rospy.loginfo('Data saved at\n' + save_path)

        rospy.loginfo('Rollout completed')

if __name__ == '__main__':
    rollout_baseline = RolloutBaseline()
    rollout_baseline.rollout()
        