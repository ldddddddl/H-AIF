from gen_rand_redbox import GenRandBox
import rospy
import time
from jetmax_control_gazebo.srv import IK, IKRequest, FK, FKRequest
from std_srvs.srv import Empty
from publisher.go_home import go_home
from multiprocessing import Process, Queue

'''
launch after pub_sign script
'''

def move_xyz(ik, tar_pos:list=[0, -150, 95], home_pos:list=[0, -150, 200], move_flag:str='xy', v:float=0.02):
    '''
    @param:
    :ik: forward_kinematics
    :tar_pos: target_position [list]
    :home_pos: home_position [list]
    :move_flag: move xy or z
    :v:move velocities  less -- > fast
    '''
    x_t, y_t, z_t = tar_pos
    x_c, y_c, z_c = home_pos
    if move_flag == 'xy' or not x_t == None or not y_t == None:
        if x_t > x_c:
            x_sign = 'positive'
        else:
            x_sign = 'negative'
        if y_t > y_c:
            y_sign = 'positive'
        else:
            y_sign = 'negative'
    elif move_flag == 'z' or not z_t == None:
        if z_t > z_c:
            z_sign = 'positive'
        else:
            z_sign = 'negative'
    else:
        raise ValueError("move_flag == xy or z")

    while True:
        if move_flag == 'xy' or not x_t == None or not y_t == None:
            if x_sign == 'positive' and x_c <= x_t:
                x_c += 1
            elif x_sign == 'negative' and x_c >= x_t:
                x_c -= 1
            else:
                pass

            if y_sign == 'positive' and y_c <= y_t:
                y_c += 1
            elif y_sign == 'negative' and y_c >= y_t:
                y_c -= 1
            else:
                pass

            if x_c >= x_t and y_c >= y_t and move_flag == 'xy' and x_sign == 'positive' and y_sign == 'positive':
                break
            elif x_c <= x_t and y_c <= y_t and move_flag == 'xy' and x_sign == 'negative' and y_sign == 'negative':
                break
            elif x_c >= x_t and y_c <= y_t and move_flag == 'xy' and x_sign == 'positive' and y_sign == 'negative':
                break
            elif x_c <= x_t and y_c >= y_t and move_flag == 'xy' and x_sign == 'negative' and y_sign == 'positive':
                break

        elif move_flag == 'z' or not z_t == None:

            if z_sign == 'positive' and z_c <= z_t:
                z_c += 1
            elif z_sign == 'negative' and z_c >= z_t:
                z_c -= 1
            else:
                pass

            if z_c >= z_t and move_flag == 'z' and z_sign == 'positive':
                break
            elif z_c <= z_t and move_flag == 'z' and z_sign == 'negative':
                break
        else:
            raise ValueError("move_flag == xy or z")
        ik(IKRequest(x_c, y_c, z_c))
        rospy.sleep(v)
    return [x_c, y_c, z_c]



def monitor_function_call(func, joint_read):
    def wrapper(*args, **kwargs):
        if joint_read == None:
            raise ValueError("pls delivery a joint ")
        timestamp = joint_read.get_timestamp()
        joint_read.sucker_act()
        # call_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        rospy.loginfo(f"Function {func.resolved_name} called at {timestamp}")
        func_results = func(*args, **kwargs)
        return func_results, timestamp
    return wrapper


def move_to(*arg, final_pos=None, is_gohome=False):
    '''
    :@param
    :x, y, z: object generated pos
    :r: object r
    :x_offset, y_offset, z_offset: obj pos offset
    :pd_pos_x, pd_pos_y, pd_pos_z: putdown pos

    '''
    pos_list = arg
    try:
        rospy.init_node("simple_pick")
    except:
        pass
    ik = rospy.ServiceProxy("/jetmax_control/inverse_kinematics", IK)
    # fk = rospy.ServiceProxy("/jetmax_control/forward_kinematics", FK)
    # on = rospy.ServiceProxy("/jetmax/vacuum_gripper/on", Empty)
    # off = rospy.ServiceProxy("/jetmax/vacuum_gripper/off", Empty)
    # on = monitor_function_call(rospy.ServiceProxy("/jetmax/vacuum_gripper/on", Empty), joint_read=joint_read)
    # off = monitor_function_call(rospy.ServiceProxy("/jetmax/vacuum_gripper/off", Empty), joint_read=joint_read)
    if is_gohome:
        go_home()
    # time.sleep(1)
    # print("========START========")
    '''
    action seq:  xy --> -z --> +z --> xy --> -z --> +z --> go_home
    sucker z offset: 61
    '''
    if final_pos is None:
        curr_pos = [0, -150, 200]
    else:
        curr_pos = final_pos
    r_offset = 60
    sucker_flag = None
    for x, y, z in pos_list[0]:
        # cam_1 = Process(target=move_xyz, args=(fk, x, y, z,))
        curr_pos = move_xyz(ik, [x, y, None], curr_pos,  move_flag='xy')
        time.sleep(1)
        curr_pos = move_xyz(ik, [None, None, z], curr_pos,  move_flag='z')
        time.sleep(1)
    go_home()


if __name__ == "__main__":
    points = [
        [0, -190, 100],
        [0, -170, 100],
        [0, -190, 140],
        [0, -140, 100],
        [0, -180, 100],
        [0, -150, 140],
    ]
    move_to(points)