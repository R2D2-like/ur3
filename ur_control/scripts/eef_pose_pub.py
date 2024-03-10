#!/usr/bin/env python3
from ur_control.arm import Arm
import rospy
import numpy as np
from geometry_msgs.msg import Pose

class EefPosePub:
    def __init__(self) -> None:
        rospy.init_node('eef_pose_pub', anonymous=True)
        self.arm = Arm(gripper_type=None, ee_link='wrist_3_link')
        rospy.loginfo('EefPosePub node initialized')

        # Publisher
        self.pub_eef_pose = rospy.Publisher('/eef_pose', Pose, queue_size=10)

    def main(self):
        rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            current_pos = self.arm.end_effector()
            pose = Pose()
            pose.position.x = current_pos[0]
            pose.position.y = current_pos[1]
            pose.position.z = current_pos[2]
            pose.orientation.x = current_pos[3]
            pose.orientation.y = current_pos[4]
            pose.orientation.z = current_pos[5]
            pose.orientation.w = current_pos[6]
            self.pub_eef_pose.publish(pose)
            rate.sleep()

if __name__ == '__main__':
    eef_pose_pub = EefPosePub()
    eef_pose_pub.main()