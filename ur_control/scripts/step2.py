#!/usr/bin/env python3
from ur_control.arm import Arm
import rospy
import numpy as np
import os

class Step2:
    def __init__(self) -> None:
        rospy.init_node('step2', anonymous=True)
        self.arm = Arm(gripper_type=None, ee_link='wrist_3_link') # with gripper
        self.data = None
        self.save_dir = '/root/Research_Internship_at_GVlab/real_exp/step2/'
        self.save_name = input('Enter the name of the save file: ')
        rospy.loginfo('Step2 node initialized')

    def record(self):
        rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            if self.data is None:
                self.data = self.arm.end_effector()
            else:
                self.data = np.vstack((self.data, self.arm.end_effector()))
            if self.data.shape[0] >= 2000:
                ft_data = self.arm.get_wrench_history(hist_size=2000)
                break
            rate.sleep()
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        np.save(self.save_dir + self.save_name + '_traj.npy', self.data) # (2000, 3)
        np.save(self.save_dir + self.save_name + '_traj_ft.npy', np.concatenate((self.data, ft_data), axis=1)) # (2000, 9)
        rospy.loginfo('Data saved')

if __name__ == '__main__':
    step2 = Step2()
    step2.record()

