#!/usr/bin/env python3
from ur_control.arm import Arm
import rospy
import numpy as np
import os
from ur_control import transformations, traj_utils, conversions


class ExploratoryAction:
    def __init__(self):
        rospy.init_node('exploratory_action', anonymous=True)
        self.arm = Arm(gripper_type=None, ee_link='wrist_3_link') # with gripper
        self.arm.set_ft_filtering()
        self.arm.zero_ft_sensor()
        self.pressing_data = None
        mode = input('0:step1, 1:rollout: ')
        stiffness = input('stiffness level (1, 2, 3, 4): ')
        friction = input('friction level (1, 2, 3): ')
        self.sponge = 's' + stiffness + 'f' + friction
        rospy.loginfo(self.sponge)
        if mode == '0':
            self.save_dir = '/root/Research_Internship_at_GVlab/real/step1/data/'
            trial = input('Trial (1, 2, 3, 4, 5, 6, 7, 8): ')
            self.save_name = self.sponge + '_' + trial + '.npz'
        else:
            self.save_dir = '/root/Research_Internship_at_GVlab/real/rollout/data/exploratory/'
            self.save_name = self.sponge + '.npz'

        rospy.loginfo('Pressing node initialized')
        input('Press Enter to start!')
        rospy.loginfo('Start Pressing...')

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

    def pressing(self):
        self.arm.zero_ft_sensor()
        self.move_endeffector([0, 0, 0.02, 0, 0, 0], target_time=2)
        self.pressing_data = self.arm.get_wrench_history(hist_size=200)

    def save_pressing_data(self):
        save_dir = self.save_dir + 'pressing/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print('data shape: ', self.pressing_data.shape)

        # キーワード引数を辞書として定義
        kwargs = {self.sponge: self.pressing_data}

        # 辞書をアンパックしてnp.savezに渡す
        save_path = save_dir + self.save_name
        np.savez(save_path, **kwargs) # (200, 6)
        rospy.loginfo('Data saved at\n' + save_path)

    def lateral_movements(self):
        self.move_endeffector([-0.05, -0.015, -0.001, 0, 0, 0], target_time=1)
        self.lateral_movements_data = self.arm.get_wrench_history(hist_size=100)
        rospy.sleep(1)
        self.move_endeffector([0.05, 0.015, 0.001, 0, 0, 0], target_time=1)

        self.lateral_movements_data = np.concatenate((self.lateral_movements_data, self.arm.get_wrench_history(hist_size=100))) 

    def save_lateral_data(self):
        save_dir = self.save_dir + 'lateral/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print('data shape: ', self.lateral_movements_data.shape)

        # キーワード引数を辞書として定義
        kwargs = {self.sponge: self.lateral_movements_data}

        # 辞書をアンパックしてnp.savezに渡す
        save_path = save_dir + self.save_name
        np.savez(save_path, **kwargs) # (200, 6)
        rospy.loginfo('Data saved at\n' + save_path)

    def going_up(self):
        self.move_endeffector([0, 0, -0.025, 0, 0, 0], target_time=2)




if __name__ == '__main__':
    is_sim = input('Is this a simulation? (y/n): ')
    if is_sim == 'y':
        rospy.loginfo('This is a simulation')
        is_sim = True
    else:
        rospy.loginfo('This is a real robot')
        is_sim = False
    exp = ExploratoryAction()
    if is_sim:
        exp.go_to_initial_pose()
    exp.pressing()
    exp.save_pressing_data()
    exp.going_up()
    # 待機
    input('Press Enter to start lateral movements')
    exp.lateral_movements()
    exp.save_lateral_data()
    rospy.loginfo('Pressing completed')

        