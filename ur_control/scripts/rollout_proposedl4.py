#!/usr/bin/env python3
from ur_control.arm import Arm
import rospy
import numpy as np
import os
import sys
# add /root/Research_Internship_at_GVlab/scripts/config
sys.path.append('/root/Research_Internship_at_GVlab/scripts/config')
from values import SCALING_FACTOR, DEMO_TRAJECTORY_MIN, DEMO_TRAJECTORY_MAX, DEMO_FORCE_TORQUE_MIN, DEMO_FORCE_TORQUE_MAX, DEMO_Z_DIFF_MIN, DEMO_Z_DIFF_MAX
import torch
sys.path.append('/root/Research_Internship_at_GVlab/scripts/train/')
from lfd_proposed import LfDProposed
from ur_control import transformations
from geometry_msgs.msg import Pose
from geometry_msgs.msg import WrenchStamped
import collections
from lfd_baseline import LfDBaseline


class RolloutProposed:
    def __init__(self) -> None:
        rospy.init_node('rollout', anonymous=True)
        self.arm = Arm(gripper_type=None, ee_link='wrist_3_link') # with gripper
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Subscriber
        self.sub_eef_pose = rospy.Subscriber('/eef_pose', Pose, self.eef_pose_callback)
        self.sub_ft = rospy.Subscriber('/wrench/filtered', WrenchStamped, self.ft_callback)

        # first in first out
        self.eef_pose_history = collections.deque(maxlen=2000)
        self.ft_history = collections.deque(maxlen=2000)
        # 0で初期化
        self.eef_pose_history.extend(np.zeros((2000, 7))) # (2000, 7)
        self.ft_history.extend(np.zeros((2000, 6))) # (2000, 6)

        self.init_eef_position_history = []
        self.init_ft_history = []

        self.base_dir = '/root/Research_Internship_at_GVlab/real/'
        stiffness = input('stiffness level (1, 2, 3, 4): ')
        friction = input('friction level (1, 2, 3): ')
        self.sponge = 's' + stiffness + 'f' + friction
        self.base_save_dir = self.base_dir + 'rollout/data/'

        self.vae_inputs = np.load(self.base_dir + 'rollout/data/exploratory/exploratory_action_preprocessed.npz')[self.sponge] # normalized

        model_weights_path = self.base_dir + 'model/proposed/proposed_model2.pth'
        self.lfd = LfDProposed(tcn_input_size=6, tcn_output_size=7, mlp_output_size=1).to(self.device)
        self.lfd.load_state_dict(torch.load(model_weights_path))
        self.lfd.eval()
        rospy.loginfo('Rollout node initialized')

    def eef_pose_callback(self, msg):
        self.current_pos = np.array([msg.position.x, msg.position.y, msg.position.z, \
                                msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        self.eef_pose_history.append(self.current_pos) # (7, 2000)

    def ft_callback(self, msg):
        self.current_ft = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z, \
                                msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])
        self.ft_history.append(self.current_ft) # (6, 2000)


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

    def init_pressing(self):
        self.arm.zero_ft_sensor()
        print('aaaaaaa')
        self.move_endeffector([0, 0, 0.01, 0, 0, 0], target_time=2)
        print('bbbbbb')

    def output2position(self, normalized_output):
        # 正規化された出力をもとの値に戻す
        # normalized_output = np.load(self.base_dir + self.save_name) #(2000, 3)
        output = normalized_output/SCALING_FACTOR
        output = normalized_output * (np.array(DEMO_TRAJECTORY_MAX) - np.array(DEMO_TRAJECTORY_MIN)) + np.array(DEMO_TRAJECTORY_MIN)
        # idx2だけもとの値に戻す
        output[:,2] = normalized_output[:,2]
        return output
    
    def position2input(self, position):
        # 位置をモデルの入力に変換
        print(np.array(position).shape, np.array(DEMO_TRAJECTORY_MIN)[:3].shape)
        input = (np.array(position) - np.array(DEMO_TRAJECTORY_MIN)[:3]) / (np.array(DEMO_TRAJECTORY_MAX)[:3] - np.array(DEMO_TRAJECTORY_MIN)[:3])
        input *= SCALING_FACTOR
        self.last_z = input[-1][2]
        input = input[:,:2]
        return input
    
    def ft2input(self, ft):
        # 力覚センサの値をモデルの入力に変換
        input = (ft - np.array(DEMO_FORCE_TORQUE_MIN)) / (np.array(DEMO_FORCE_TORQUE_MAX) - np.array(DEMO_FORCE_TORQUE_MIN))
        return input
    
    def predict(self, vae_inputs, tcn_inputs):
        # inference
        z_diff = self.lfd(vae_inputs, tcn_inputs)
        z_diff = z_diff.detach().cpu().numpy()[0]
        # 正規化された出力をもとの値に戻す
        z_diff = z_diff/SCALING_FACTOR
        z_diff = z_diff * (np.array(DEMO_Z_DIFF_MAX) - np.array(DEMO_Z_DIFF_MIN)) + np.array(DEMO_Z_DIFF_MIN)
        z = self.last_z + z_diff
        # 位置に変換
        z /= SCALING_FACTOR
        z *= (np.array(DEMO_TRAJECTORY_MAX)[2] - np.array(DEMO_TRAJECTORY_MIN))[2] + np.array(DEMO_TRAJECTORY_MIN)[2]
        return z
    
    def predict_eef_position(self):
        #device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # load data
        data = np.load(self.base_dir + 'rollout/data/exploratory/exploratory_action_preprocessed.npz')[self.sponge]  # normalized
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
        eef_position = self.output2position(output)
        save_dir = self.base_save_dir + 'baseline/predicted/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.savez(save_dir + self.sponge + '.npz', eef_position=eef_position)
        print('Data saved at\n: ', save_dir + self.sponge + '.npz')
        rospy.loginfo('Inference completed')
        return eef_position[0]  # (2000, 3)


    def rollout(self):
        start_time = rospy.Time.now()
        # initialize motion
        self.init_pressing()
        pred_eef_position_history = []
        # self.eef_pose_history[:,len(self.init_eef_position_history):] = self.init_eef_position_history
        # ft_history = np.zeros_like(6, 2000)
        # ft_history[:,len(self.init_ft_history):] = self.init_ft_history

        ee_position = self.predict_eef_position() #(2000, 3)
        print(ee_position.shape)
        target_time = 0.2
        self.last_z = self.current_pos[2]
        for i in range(1500, 2000, 20):
            print(i)
            pose_goal = self.arm.end_effector()
            pose_goal[0] = ee_position[i, 0]
            pose_goal[1] = ee_position[i, 1]
            ft_history = np.array(self.arm.get_wrench_history(hist_size=100))[::20] # (5, 6)
            ft_history = self.ft2input(ft_history) # (5, 6)
            # (5, 6) -> (6, 5)
            tcn_inputs = np.expand_dims(ft_history.T, axis=0) # (1, 6, 5)
            print(self.vae_inputs.shape, tcn_inputs.shape)
            ee_position[i, 2] = self.predict(torch.tensor(self.vae_inputs).float().to(self.device), torch.tensor(tcn_inputs).float().to(self.device))
            print('next_eef_position', ee_position[i])
            self.last_z = self.current_pos[2]
            pose_goal[2] = ee_position[i, 2]
            try:
                self.arm.set_target_pose(pose=pose_goal, wait=True, target_time=target_time)
            except Exception as e:
                print(e)
            self.eef_pose_history.append(self.current_pos)


        traj_history = self.eef_pose_history #[-2000:] # (2000, 7)
        ft_history = self.arm.get_wrench_history(hist_size=2000) # (2000, 6)
        save_dir = self.base_save_dir + 'baseline/result/' 
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = save_dir + self.sponge + '.npz'
        np.savez(save_path, pose=traj_history, ft=ft_history)
        rospy.loginfo('Data saved at\n' + save_path)

        rospy.loginfo('Rollout completed')

if __name__ == '__main__':
    rollout_proposed = RolloutProposed()
    rollout_proposed.rollout()
        