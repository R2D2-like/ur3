#!/usr/bin/env python3
from ur_control.arm import Arm
from ur_control.impedance_control import AdmittanceModel
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
from ur_control import transformations


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

        self.base_dir = '/root/Research_Internship_at_GVlab/real/'

    def eef_pose_callback(self, msg):
        self.current_pos = np.array([msg.position.x, msg.position.y, msg.position.z, \
                                msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        
    def ft_callback(self, msg):
        self.current_ft = np.array([msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z, \
                                msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])
        self.ft_history.append(self.current_ft) # (6, 2000)

    def init_pressing(self):
        self.arm.zero_ft_sensor()
        print('aaaaaaa')
        self.move_endeffector([0, 0, 0.01, 0, 0, 0], target_time=2)
        print('bbbbbb')

    def output2position(self, normalized_output):
        # 正規化された出力をもとの値に戻す
        # normalized_output = np.load(self.base_dir + self.save_name) #(2000, 3)
        normalized_output /= SCALING_FACTOR
        output = normalized_output * (np.array(DEMO_TRAJECTORY_MAX) - np.array(DEMO_TRAJECTORY_MIN)) + np.array(DEMO_TRAJECTORY_MIN)
        return output
    
    
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
    
    def init_admittance_control(self):
        # 各パラメータを設定
        inertia = 0.5  # 慣性
        stiffness = 15.0  # 剛性
        damper = 5#15  # ダンパー
        # inertia = np.array([[6, 0, 0, 0, 0, 0], [0, 6, 0, 0, 0, 0], [0, 0, 6, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0.5]])
        # damper = np.array([[60, 0, 0, 0, 0, 0], [0, 60, 0, 0, 0, 0], [0, 0, 60, 0, 0, 0], [0, 0, 0, 15, 0, 0], [0, 0, 0, 0, 15, 0], [0, 0, 0, 0, 0, 15]])
        # stiffness = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
        dt = 0.2  # サンプリング時間
        method = "traditional"  # 3つのうちの1つを選択
        # method = "integration"

        # admittance modelを作成
        admittance = AdmittanceModel(inertia, stiffness, damper, dt, method)
        # admittance modelをリセット
        admittance.reset()

        return admittance
    
    def impedance_control(self, admittance, pose_goal, fz, kp=0.01):
        # print('self.eef_ft_history:', self.eef_ft_history)
        # fz = self.eef_ft_history[-1][2]
        print('fz:', fz)
        deltax = admittance.control(fz)
        print('deltax:', -deltax*kp)
        pose_goal[2] -= deltax * kp
        print('z', pose_goal[2])

        return pose_goal

    def rollout(self):
        rospy.sleep(1)
        # self.init_pressing()
        self.eef_pose_history =[]
        self.eef_ft_history = []
        target_time = 0.02
        current_x = self.current_pos[0] 
        current_y = self.current_pos[1] 
        current_z = self.current_pos[2] 

        admittance = self.init_admittance_control()
        for i in range(1500, 10000, 20):
            print(i)
            pose_goal = self.arm.end_effector()
            pose_goal[0] = current_x
            pose_goal[1] = current_y
            pose_goal[2] = self.current_pos[2] 
            fz = np.array(self.arm.get_wrench_history(hist_size=100))[::20][-1][2]
            pose_goal = self.impedance_control(admittance, pose_goal, fz)

            try:
                self.arm.set_target_pose(pose=pose_goal, wait=True, target_time=target_time)
            except Exception as e:
                print(e)
            self.eef_pose_history.append(self.current_pos)
            self.eef_ft_history.append(self.current_ft)
            

        rospy.loginfo('Impedance control test completed')

if __name__ == '__main__':
    rollout_baseline = RolloutBaseline()
    rollout_baseline.rollout()
        