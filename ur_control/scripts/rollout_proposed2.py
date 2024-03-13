#!/usr/bin/env python3
from ur_control.arm import Arm
import rospy
import numpy as np
import os
import sys
# add /root/Research_Internship_at_GVlab/scripts/config
sys.path.append('/root/Research_Internship_at_GVlab/scripts/config')
from values import SCALING_FACTOR, DEMO_TRAJECTORY_MIN, DEMO_TRAJECTORY_MAX, DEMO_FORCE_TORQUE_MIN, DEMO_FORCE_TORQUE_MAX
import torch
sys.path.append('/root/Research_Internship_at_GVlab/scripts/train/')
from lfd_proposed import LfDProposed
from ur_control import transformations
from geometry_msgs.msg import Pose
from geometry_msgs.msg import WrenchStamped
import collections


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
        self.lfd = LfDProposed(tcn_input_size=8).to(self.device)
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
        self.last_z = position[-1][2]
        # 位置をモデルの入力に変換
        print(np.array(position).shape, np.array(DEMO_TRAJECTORY_MIN)[:3].shape)
        input = (np.array(position) - np.array(DEMO_TRAJECTORY_MIN)[:3]) / (np.array(DEMO_TRAJECTORY_MAX)[:3] - np.array(DEMO_TRAJECTORY_MIN)[:3])
        input *= SCALING_FACTOR
        input = input[:,:2]
        return input
    
    def ft2input(self, ft):
        # 力覚センサの値をモデルの入力に変換
        input = (ft - np.array(DEMO_FORCE_TORQUE_MIN)) / (np.array(DEMO_FORCE_TORQUE_MAX) - np.array(DEMO_FORCE_TORQUE_MIN))
        return input
    
    def predict(self, vae_inputs, tcn_inputs):
        # inference
        output = self.lfd(vae_inputs, tcn_inputs)
        output = output.detach().cpu().numpy()
        output[0][2] += self.last_z
        print('force',tcn_inputs[0][2][1999])
        print('diff', output[0][2])
        eef_position = self.output2position(output)
        return eef_position[0] #(3,)


    def rollout(self):
        start_time = rospy.Time.now()
        # initialize motion
        self.init_pressing()
        pred_eef_position_history = []
        # self.eef_pose_history[:,len(self.init_eef_position_history):] = self.init_eef_position_history
        # ft_history = np.zeros_like(6, 2000)
        # ft_history[:,len(self.init_ft_history):] = self.init_ft_history



        while (rospy.Time.now() - start_time).to_sec() < 20:
            ft_history = np.array(self.ft_history) # (2000, 6)
            eef_position_history = np.array(self.eef_pose_history)[:,:3] # (2000, 3)
            ft_history = self.ft2input(ft_history) # (2000, 6)
            eef_position_history = self.position2input(eef_position_history) # (2000, 2) #np.array(a[:-3])[:,:3]
            tcn_inputs = np.concatenate([eef_position_history, ft_history], axis=1) # (2000, 8)
            # (100, 9) -> (9, 100)
            tcn_inputs = np.expand_dims(tcn_inputs.T, axis=0)
            print(self.vae_inputs.shape, tcn_inputs.shape)
            next_eef_position = self.predict(torch.tensor(self.vae_inputs).float().to(self.device), torch.tensor(tcn_inputs).float().to(self.device))
            pred_eef_position_history.append(next_eef_position)
            print('next_eef_position', next_eef_position)
            
            target_time = 0.5

            # for i in range(next_eef_position.shape[0]):
            pose_goal = self.arm.end_effector()
            pose_goal[0] = next_eef_position[0]
            pose_goal[1] = next_eef_position[1]
            pose_goal[2] = next_eef_position[2]
            try:
                self.arm.set_target_pose(pose=pose_goal, wait=True, target_time=target_time)
            except Exception as e:
                print(e)
            

        # save data
        pred_history = next_eef_position#[-2000:]
        save_dir = self.base_save_dir + 'proposed/predicted/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = save_dir + self.sponge + '.npz'
        np.savez(save_path, eef_position=pred_history)
        rospy.loginfo('Data saved at\n' + save_path)

        traj_history = self.eef_pose_history[-2000:] # (2000, 7)
        ft_history = self.arm.get_wrench_history(hist_size=2000) # (2000, 6)
        save_dir = self.base_save_dir + 'proposed/results/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = save_dir + self.sponge + '.npz'
        np.savez(save_path, pose=traj_history, ft=ft_history)
        rospy.loginfo('Data saved at\n' + save_path)

        rospy.loginfo('Rollout completed')

if __name__ == '__main__':
    rollout_proposed = RolloutProposed()
    rollout_proposed.rollout()
        