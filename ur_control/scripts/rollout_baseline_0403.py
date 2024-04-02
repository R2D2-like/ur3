#!/usr/bin/env python3
from ur_control.arm import Arm
import rospy
import numpy as np
import os
import sys
# add /root/Research_Internship_at_GVlab/scripts/config
sys.path.append('/root/Research_Internship_at_GVlab/scripts/config')
from values import SCALING_FACTOR, DEMO_TRAJECTORY_MIN, DEMO_TRAJECTORY_MAX
import torch
from geometry_msgs.msg import Pose, WrenchStamped
sys.path.append('/root/Research_Internship_at_GVlab/scripts/train/')
from lfd_baseline import LfDBaseline
import collections


class RolloutBaseline:
    def __init__(self) -> None:
        rospy.init_node('rollout_baseline', anonymous=True)
        self.arm = Arm(gripper_type=None, ee_link='wrist_3_link') # with gripper

        # Subscriber
        self.sub_eef_pose = rospy.Subscriber('/eef_pose', Pose, self.eef_pose_callback)
        self.sub_ft = rospy.Subscriber('/wrench/filtered', WrenchStamped, self.ft_callback)
        # self.eef_pose_history = []

        # first in first out
        self.eef_pose_history = collections.deque(maxlen=100)
        self.ft_history = collections.deque(maxlen=100)

        self.base_dir = '/root/Research_Internship_at_GVlab/data0403/real/'
        stiffness = input('stiffness level (1, 2, 3, 4): ')
        friction = input('friction level (1, 2, 3): ')
        self.sponge = 's' + stiffness + 'f' + friction
        self.height = input('low high slope:')
        self.base_save_dir = '/root/Research_Internship_at_GVlab/data0403/real/rollout/data/'
        rospy.loginfo('Rollout node initialized')

    def eef_pose_callback(self, msg):
        self.current_pos = np.array([msg.position.x, msg.position.y, msg.position.z, \
                                msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        
    def ft_callback(self, msg):
        self.current_ft = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z, \
                                msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])
        self.ft_history.append(self.current_ft) # (6, 2000)

    def output2position(self, normalized_output):
        # 正規化された出力をもとの値に戻す
        # normalized_output = np.load(self.base_dir + self.save_name) #(2000, 3)
        normalized_output /= SCALING_FACTOR
        output = normalized_output * (np.array(DEMO_TRAJECTORY_MAX) - np.array(DEMO_TRAJECTORY_MIN)) + np.array(DEMO_TRAJECTORY_MIN)
        return output
    
    def predict_eef_position(self):
        #device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # load data
        # data = np.load(self.base_dir + 'rollout/data/exploratory/exploratory_action_preprocessed.npz')['s1f1']  # normalized
        data = np.load('/root/Research_Internship_at_GVlab/data0403/real/step1/data/exploratory_action_preprocessed.npz')[self.sponge]  # normalized

        # instantiate the model
        model = LfDBaseline(input_dim=6, output_dim=3, latent_dim=5, hidden_dim=32).to(device)
        # load model weights
        model_weights_path = self.base_dir + 'model/baseline/baseline_model.pth'
        state_dict = torch.load(model_weights_path)
        model.load_state_dict(state_dict, strict=False)  # Load the state dict
        model.eval()  # Now this should work as model is properly instantiated
        # inference
        output = model(torch.tensor(data).float().to(device))  # Ensure data is in the correct dtype for the model
        output = output.detach().cpu().numpy()
        eef_position = self.output2position(output)#(1,2000,3)
        save_dir = self.base_save_dir + 'baseline/predicted/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = save_dir + self.sponge + '_' + self.height +'.npz'
        np.savez(save_path, eef_position=eef_position[0])#[1500::20,:]) #(25,3)
        print('Data saved at\n: ', save_path)
        rospy.loginfo('Inference completed')

        return eef_position[0]  # (2000, 3)

    def rollout(self):
        # rospy.sleep(5)
        ee_position = self.predict_eef_position() #(2000, 3)
        print(ee_position.shape)
        self.eef_pose_history =[]
        self.eef_ft_history = []
        target_time = 0.02
        for i in range(ee_position.shape[0]):
            print(i)
            print(ee_position[i])
            pose_goal = self.arm.end_effector()
            pose_goal[0] = ee_position[i, 0]
            pose_goal[1] = ee_position[i, 1]
            pose_goal[2] = ee_position[i, 2]
            try:
                self.arm.set_target_pose(pose=pose_goal, wait=True, target_time=target_time)
            except Exception as e:
                print(e)
            self.eef_pose_history.append(self.current_pos)
            self.eef_ft_history.append(self.current_ft)
            

        traj_history = self.eef_pose_history #[-2000:] # (2000, 7)
        ft_history = self.eef_ft_history
        save_dir = self.base_save_dir + 'baseline/result/' 
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # save_path = save_dir + self.sponge + '.npz'
        save_path = save_dir + self.sponge + '_' + self.height +'.npz'
        np.savez(save_path, pose=traj_history, ft=ft_history)
        rospy.loginfo('Data saved at\n' + save_path)

        rospy.loginfo('Rollout completed')

if __name__ == '__main__':
    rollout_baseline = RolloutBaseline()
    rollout_baseline.rollout()
        