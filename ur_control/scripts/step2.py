#!/usr/bin/env python3
from ur_control.arm import Arm
import rospy
import numpy as np
import os
import collections
from geometry_msgs.msg import Pose
from scipy.signal import butter, sosfilt

class Step2:
    def __init__(self) -> None:
        rospy.init_node('step2', anonymous=True)
        self.arm = Arm(gripper_type=None, ee_link='wrist_3_link') # with gripper

        # Subscriber
        self.sub_eef_pose = rospy.Subscriber('/eef_pose', Pose, self.eef_pose_callback)
        self.eef_pose_history = []

        self.data = None
        self.save_dir = '/root/Research_Internship_at_GVlab/data0404/real/step2/data/'
        stiffness = input('stiffness level (1, 2, 3, 4): ')
        friction = input('friction level (1, 2, 3): ')
        self.save_name = 's' + stiffness + 'f' + friction
        self.trial = input('Trial (1, 2, 3, 4, 5, 6, 7, 8): ')
        rospy.loginfo('Step2 node initialized')
        input('Press Enter to start!')
        rospy.loginfo('Start Recording...')

    def eef_pose_callback(self, msg):
        current_pos = np.array([msg.position.x, msg.position.y, msg.position.z, \
                                msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        self.eef_pose_history.append(current_pos)

    def filter(self, data):
        # データのサンプリング周波数とカットオフ周波数を設定
        fs = 3000.0  # サンプリング周波数 (Hz)
        fc = 5.0   # カットオフ周波数 (Hz)

        # バターワースローパスフィルタの設計
        sos = butter(N=4, Wn=fc/(fs/2), btype='low', output='sos')

        # フィルタリングされたデータを格納する配列を準備
        filtered_data = np.zeros_like(data) #(400, 6)

        # 各列に対してローパスフィルタを適用
        for i in range(data.shape[1]): # i=0,1,2,3,4,5
            filtered_data[:, i] = sosfilt(sos, data[:, i])

        return filtered_data

    def record(self):
        rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            if self.data is None:
                self.data = self.arm.end_effector()
            else:
                self.data = np.vstack((self.data, self.arm.end_effector()))
            if self.data.shape[0] >= 2400:
                ft_data = self.arm.get_wrench_history(hist_size=2400)
                break
            rate.sleep()
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        traj_history = self.eef_pose_history[-2400:] # (2000, 7)
        ft_history = self.arm.get_wrench_history(hist_size=2400) # (2000, 6)
        # ft_history = self.filter(ft_history)
        save_path = self.save_dir + self.save_name + '_' + self.trial + '.npz'
        np.savez(save_path, pose=traj_history, ft=ft_history)
        rospy.loginfo('Data saved')

if __name__ == '__main__':
    step2 = Step2()
    step2.record()

