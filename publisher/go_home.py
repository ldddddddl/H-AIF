#!/usr/bin/env python2.7
import rospy
from jetmax_control_gazebo.srv import FK, FKRequest, FKResponse

# def go_home():
# if __name__ == "__main__":
#     rospy.init_node("go_home")
#     go_home = rospy.ServiceProxy("/jetmax_control/forward_kinematics", FK)
#     ret = go_home(FKRequest(90, 90, 0))
#     rospy.logdebug(ret)



def go_home():
    # 初始化节点，如果节点已经初始化，这行代码不会有影响
    if not rospy.core.is_initialized():
        rospy.init_node("go_home", anonymous=True)
    
    # 定义服务代理
    go_home_service = rospy.ServiceProxy("/jetmax_control/forward_kinematics", FK)
    
    # 发送请求
    try:
        # x = 188.56, y = 0, z = 84.4
        ret = go_home_service(FKRequest(90, 90, 0))
        rospy.logdebug(ret)
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")

# 只有在直接运行该脚本时才会执行以下代码
if __name__ == "__main__":
    go_home()
