#!/usr/bin/env python2.7
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import dotmap
import sys
sys.path.append(r"/home/ubuntu/Desktop/action_generation/publisher")
import math
from rospy.topics import Publisher
from std_msgs.msg import Float64
import jetmax_kinematics
from jetmax_control_gazebo.srv import IK, IKRequest, IKResponse
import time


NUM_OF_JOINTS = 9
joints_publishers = []
fk_service = None

def publisher_callback(joint_datas):
    joint_angles = [deg * math.pi / 180 for deg in joint_datas]
    # Set the arm joint angles
    for i in range(NUM_OF_JOINTS):
        joints_publishers[i].publish(Float64(joint_angles[i]))
    return [True, ]

def joint_publisher():
    global joints_publishers, fk_service
    # if rospy.core.is_initialized():
    #     rospy.signal_shutdown("node_initialized")
    #     time.sleep(1)
    # if not rospy.core.is_initialized():
    # rospy.init_node('joint_publisher', anonymous=True)
    for i in range(NUM_OF_JOINTS):
        pub = rospy.Publisher("/jetmax/joint{}_position_controller/command".format(i + 1),Float64, queue_size=10)
        joints_publishers.append(pub)

    fk_service = rospy.Service("/jetmax_control/inverse_kinematics", IK, publisher_callback)
    rospy.loginfo("Ready to send joint angles")
    # try:
    #     rospy.spin()
    # except Exception as e:
    #     rospy.logerr(e)
    #     sys.exit(-1)
