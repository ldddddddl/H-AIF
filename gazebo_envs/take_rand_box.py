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

def move_xyz(ik, tar_pos:list=[0, -150, 95], home_pos:list=[0, -150, 200], move_flag:str='xy'):
    '''
    param:
    @ik: forward_kinematics
    @tar_pos: target_position [list]
    @home_pos: home_position [list]
    @move_flag: move xy or z
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
        rospy.sleep(0.01)
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


def take_rand_box(x, y, z=0.01, r=0.0, x_offset=0, y_offset=0, z_offset=0, pd_pos_x=0, pd_pos_y=-150, pd_pos_z=61,
                  sucker=True, joint_read=None):
    '''
    :param:
    @x, y, z: object generated pos
    @r: object r
    @x_offset, y_offset, z_offset: obj pos offset
    @pd_pos_x, pd_pos_y, pd_pos_z: putdown pos
    '''
    try:
        rospy.init_node("simple_pick")
    except:
        pass
    ik = rospy.ServiceProxy("/jetmax_control/inverse_kinematics", IK)
    # fk = rospy.ServiceProxy("/jetmax_control/forward_kinematics", FK)
    # on = rospy.ServiceProxy("/jetmax/vacuum_gripper/on", Empty)
    # off = rospy.ServiceProxy("/jetmax/vacuum_gripper/off", Empty)
    on = monitor_function_call(rospy.ServiceProxy("/jetmax/vacuum_gripper/on", Empty), joint_read=joint_read)
    off = monitor_function_call(rospy.ServiceProxy("/jetmax/vacuum_gripper/off", Empty), joint_read=joint_read)
    # go_home()
    # time.sleep(1)
    # print("========START========")
    '''
    action seq:  xy --> -z --> +z --> xy --> -z --> +z --> go_home
    sucker z offset: 61
    '''
    r_offset = 60
    sucker_flag = None
    # cam_1 = Process(target=move_xyz, args=(fk, x, y, z,))
    curr_pos = move_xyz(ik, [x + x_offset, y + y_offset, None], move_flag='xy')
    time.sleep(1)
    curr_pos = move_xyz(ik, [None, None, z + z_offset + r], curr_pos,  move_flag='z')
    time.sleep(1)
    if sucker:
        _, sucker_on_timestamp = on()
        sucker_flag = True
    time.sleep(1)
    curr_pos = move_xyz(ik, [None, None, z + 70 + r], curr_pos, move_flag='z')    # rise gripper
    time.sleep(1)
    # prevent angular interference
    if pd_pos_x < -35: 
        if 60 > abs(pd_pos_y):
            curr_pos = move_xyz(ik, [(pd_pos_x + x_offset + x) / 2, pd_pos_y*2, None], curr_pos, move_flag='xy')
        else:
            curr_pos = move_xyz(ik, [(pd_pos_x + x_offset + x) / 2, pd_pos_y, None], curr_pos, move_flag='xy')
            
        time.sleep(0.5)
    curr_pos = move_xyz(ik, [pd_pos_x + x_offset, pd_pos_y + y_offset, None], curr_pos, move_flag='xy')     # target position
    time.sleep(1)
    curr_pos = move_xyz(ik, [None, None, pd_pos_z + z_offset + r], curr_pos, move_flag='z') 
    time.sleep(1)
    if sucker:
        _, sucker_off_timestamp = off()
        sucker_flag = False
    time.sleep(1)
    curr_pos = move_xyz(ik, [None, None, z + 115], curr_pos, move_flag='z')   # rise gripper
    time.sleep(1)
    # go_home()
    final_pos = [pd_pos_x + x_offset, pd_pos_y + y_offset, z + 115]
    return sucker_on_timestamp, sucker_off_timestamp, final_pos


