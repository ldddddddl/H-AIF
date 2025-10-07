#!/usr/bin/env python2.7
import math
import math
import rospy
from rospy.topics import Publisher
from std_msgs.msg import Float64
from jetmax_control_gazebo.srv import IK, IKRequest, IKResponse
# 
# simulate
AngleRotateRange = 0, 240
AngleLeftRange = 0, 180
AngleRightRange = -20, 160

# real robot angle range
AngleRotateRange_joint2angle = -60, 60
AngleLeftRange_joint2angle = -50, 30
AngleRightRange_joint2angle = -50, 40

# test
AngleRotateRange_joint2angle = -60, 60  # joint 1
AngleLeftRange_joint2angle = -40, 40    # joint 2
AngleRightRange_joint2angle = 0, 90     # joint 3

'''
max position:
y : 216


position process:
active angles --> joints_angle
xyz --> active angles --> joints_angle

'''


L0 = 84.4
L1 = 8.14
L2 = 128.41
L3 = 138.0


def joints2angle(joint_angle):
    """
    Jetmax inverse kinematics (input in radians)
    @param joint_angle: list of joint angles in radians [joint1, joint2, ..., joint9]
    @return: motor angles [rotate, angle left, angle right] in degrees
    """
    if len(joint_angle) != 9:
        rospy.logerr("Invalid joint angle list length, expected 9 elements.")
        return None

    # 关节角度是弧度，需要转换为角度再进行计算
    joint_angle_deg = [math.degrees(angle) for angle in joint_angle]

    # 计算电机角度
    alpha1 = joint_angle_deg[0] + 90  # 电机1角度
    alpha2 = 90 - joint_angle_deg[1]  # 电机2角度
    # alpha3 = -joint_angle_deg[5]      # 电机3角度
    alpha3 = joint_angle_deg[5]      # 电机3角度

    # 检查角度是否在允许的范围内
    if (AngleRotateRange_joint2angle[0] <= alpha1 <= AngleRotateRange_joint2angle[1] and
        AngleLeftRange_joint2angle[0] <= alpha2 <= AngleLeftRange_joint2angle[1] and
        AngleRightRange_joint2angle[0] <= alpha3 <= AngleRightRange_joint2angle[1]):
        motor_angles = [alpha1, alpha2, alpha3]
        print("电机角度（单位：度）:", motor_angles)
        return motor_angles
    else:
        motor_angles = [0, 0, 0]
        return motor_angles


def forward_kinematics(angle):
    """
    Jetmax forward kinematics
    @param angle: active angles [rotate, angle left, angle right]
    @return: joint angles list
    """
    alpha1, alpha2, alpha3 = angle
    if (AngleRotateRange[0] <= alpha1 <= AngleRotateRange[1] and
        AngleLeftRange[0] <= alpha2 <= AngleLeftRange[1] and
            AngleRightRange[0] <= alpha2 <= AngleRightRange[1]):

        alpha3 = -alpha3
        joint_angle = [0] * 9

        # 3 active joints
        joint_angle[0] = alpha1 - 90  # joint1
        joint_angle[1] = 90 - alpha2  # joint2
        joint_angle[5] = alpha3  # joint6

        # 6 passive joints for display
        joint_angle[2] = 90 - (alpha2 + alpha3)
        joint_angle[3] = 90 - alpha2
        joint_angle[4] = joint_angle[1]
        joint_angle[6] = joint_angle[2]
        joint_angle[7] = alpha3
        joint_angle[8] = alpha3
        print(joint_angle)
        return joint_angle
    else:
        rospy.logerr("Infeasible angle values!, feasible range is AngleRotate: ({}), AngleLeft: ({}), AngleRight: ({})".format(
            AngleRotateRange, AngleLeftRange, AngleRightRange))
        rospy.logwarn("Requested angles are; angleRotate: {:.2f}, angleLeft: {:.2f}, angleRight: {:.2f}".format(
            alpha1, alpha2, alpha3))
        return None


def inverse_kinematics(position):
    """
    Jetmax inverse kinematics
    @param position: position [x, y , z]
    @return: active joint angles
    """
    x, y, z = position
    y = -y
    if y == 0:
        if x < 0:
            theta1 = -90
        elif x > 0:
            theta1 = 90
        else:
            rospy.logerr('Invalid coordinate x:{} y:{} z:{}'.format(x, y, z))
            return None
    else:
        theta1 = math.atan(x / y)
        theta1 = 0.0 if x == 0 else theta1 * 180.0 / math.pi
        if y < 0:
            if theta1 < 0 < x:
                theta1 = 90 + theta1
                print("A", theta1)
            elif theta1 > 0 > x:
                theta1 = 90 + (90 - (90 - theta1))
                print("B", theta1)
            else:
                pass

        x = math.sqrt(x * x + y * y) - L1
        z = z - L0

        if math.sqrt(x * x + z * z) > L2 + L3:
            return None

        alpha = math.atan(z / x) * 180.0 / math.pi
        beta = math.acos((L2 * L2 + L3 * L3 - (x * x + z * z)
                          ) / (2 * L2 * L3)) * 180.0 / math.pi
        gama = math.acos((L2 * L2 - L3 * L3 + (x * x + z * z)) /
                         (2 * L2 * math.sqrt(x * x + z * z))) * 180.0 / math.pi

        pos1 = theta1 + 90
        theta2 = alpha + gama
        pos2 = 180 - theta2
        theta3 = beta + theta2
        pos3 = 180 - theta3
        return pos1, pos2, pos3
