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
from lfd_proposed import LfdProposed
from ur_control import transformations
from geometry_msgs.msg import Pose

class RolloutProposed:
    def __init__(self) -> None:
        rospy.init_node('rollout', anonymous=True)
        self.arm = Arm(gripper_type=None, ee_link='wrist_3_link') # with gripper
        
        # Subscriber
        self.sub_eef_pose = rospy.Subscriber('/eef_pose', Pose, self.eef_pose_callback)
        self.eef_pose_history = []

        self.base_dir = '/root/Research_Internship_at_GVlab/real/'
        stiffness = input('stiffness level (1, 2, 3, 4): ')
        friction = input('friction level (1, 2, 3): ')
        self.sponge = 's' + stiffness + 'f' + friction
        self.base_save_dir = self.base_dir + 'rollout/data/'

        self.vae_inputs = np.load(self.base_dir + 'rollout/data/exploratory/exploratory_action_preprocessed.npz')[self.sponge] # normalized

        model_weights_path = self.base_dir + 'model/proposed/proposed_model.pth'
        self.lfd = LfdProposed()
        self.lfd.load_state_dict(torch.load(model_weights_path))
        self.lfd.eval()
        rospy.loginfo('Rollout node initialized')

    def eef_pose_callback(self, msg):
        current_pos = np.array([msg.position.x, msg.position.y, msg.position.z, \
                                msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        self.eef_pose_history.append(current_pos)


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
        self.move_endeffector([0, 0, 0.01, 0, 0, 0], target_time=1)

    def output2position(self, normalized_output):
        # 正規化された出力をもとの値に戻す
        # normalized_output = np.load(self.base_dir + self.save_name) #(2000, 3)
        normalized_output /= SCALING_FACTOR
        output = normalized_output * (DEMO_TRAJECTORY_MAX - DEMO_TRAJECTORY_MIN) + DEMO_TRAJECTORY_MIN
        return output
    
    def position2input(self, position):
        # 位置をモデルの入力に変換
        input = (position - DEMO_TRAJECTORY_MIN) / (DEMO_TRAJECTORY_MAX - DEMO_TRAJECTORY_MIN)
        input *= SCALING_FACTOR
        return input
    
    def ft2input(self, ft):
        # 力覚センサの値をモデルの入力に変換
        input = (ft - DEMO_FORCE_TORQUE_MIN) / (DEMO_FORCE_TORQUE_MAX - DEMO_FORCE_TORQUE_MIN)
        return input
    
    def predict(self, vae_inputs, tcn_inputs):
        # inference
        output = self.lfd(vae_inputs, tcn_inputs)
        output = output.detach().numpy()
        eef_position = self.output2position(output)
        return eef_position


    def rollout(self):
        start_time = rospy.Time.now()
        # initialize motion
        self.init_pressing()

        while (rospy.Time.now() - start_time).to_sec() < 20:
            ft_history = self.arm.get_wrench_history(hist_size=100)
            ft_history = self.ft2input(ft_history) # (100, 6)
            eef_position_history = self.position2input(self.eef_pose_history[-100:][:3]) # (100, 3)
            tcn_inputs = np.concatenate([ft_history, eef_position_history], axis=1) # (100, 9)
            # (100, 9) -> (9, 100)
            tcn_inputs = tcn_inputs.T
            next_eef_position = self.predict(self.vae_inputs, tcn_inputs)
            
            target_time = 0.01

            for i in range(next_eef_position.shape[0]):
                pose_goal = self.arm.end_effector()
                pose_goal[0] = next_eef_position[i, 0]
                pose_goal[1] = next_eef_position[i, 1]
                pose_goal[2] = next_eef_position[i, 2]
                try:
                    self.arm.set_target_pose(pose=pose_goal, wait=True, target_time=target_time)
                except Exception as e:
                    print(e)

        # save data
        traj_history = self.eef_pose_history[-2000:] # (2000, 7)
        ft_history = self.arm.get_wrench_history(hist_size=2000) # (2000, 6)
        save_dir = self.base_save_dir + 'result/proposed/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = save_dir + self.sponge + '.npz'
        np.savez(save_path, pose=traj_history, ft=ft_history)
        rospy.loginfo('Data saved at\n' + save_path)

        rospy.loginfo('Rollout completed')

if __name__ == '__main__':
    rollout_proposed = RolloutProposed()
    rollout_proposed.rollout()
        