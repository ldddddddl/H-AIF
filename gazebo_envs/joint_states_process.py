#!/usr/bin/env python3
# import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
import dotmap
import sys
sys.path.append(r"/home/ubuntu/Desktop/action_generation")
from std_msgs.msg import Time
from std_msgs.msg import Bool
import os
import pandas as pd
from datetime import datetime

JOINT_STATE = dotmap.DotMap({
    "time":{
        "secs":0,
        "nsecs":0,
    },
    "positions": [0.0, 0.0, 0.0, 
                  0.0, 0.0, 0.0, 
                  0.0, 0.0, 0.0],
    "velocities": [0.0012942543145660441, 0.004554460721158058, 0.011198398455276518, 
                   0.019746380012479525, -0.00718935187185441, -0.08737472680465648, 
                   0.11094508297635947, -0.0329330156078223, -0.0312977761584206],
    "efforts": [3.257103866616262e-05, 0.0038966914294391586, 0.0, 
                3.5266609389239534e-05, -0.0014254503274990782, -0.005672463917549386, 
                0.0, -0.0003329509675893405, -0.00032880105797783443]
})

# JOINT_STATE = dotmap.DotMap()

class JointStateRead:
    def __init__(self, subscipt_path='/jetmax/joint_states', is_record:bool=True, config=None):
        try:
            rospy.init_node('read_joints_state', anonymous=True)
        except:
            pass
        rospy.Subscriber(subscipt_path, JointState, self.joint_state_callback)
        self.time_sub = rospy.Subscriber('/timestamp', Time, self.time_callback)
        self.start_sub = rospy.Subscriber('/recorddatasets/start', Bool, self.start_callback)
        self.stop_sub = rospy.Subscriber('/recorddatasets/stop', Bool, self.stop_callback)
        self.is_record = is_record  
        self.pos_dict = {}
        self.eff_dict = {}
        self.vel_dict = {}
        self.sucker_dict = {}
        self.config = config
        self.start_record_msg = False
        self.root_path = './datasets/'
        self.record_f = rospy.Rate(10)
        rospy.loginfo("Joint_datas ready!")
        self.sucker_act_flag = False
        self.sucker_act_log = 'off'

    def joint_state_callback(self, msg):
        global JOINT_STATE
        JOINT_STATE.positions = list(msg.position)
        JOINT_STATE.velocities = list(msg.velocity)
        JOINT_STATE.efforts = list(msg.effort)
        JOINT_STATE.time.secs = msg.header.stamp.secs
        JOINT_STATE.time.nsecs = msg.header.stamp.nsecs
        if self.is_record and self.start_record_msg:
            self.timestamp = int(msg.header.stamp.secs *1000 + msg.header.stamp.nsecs/1000000)
            # rospy.loginfo("timestamp:{}".format(timestamp))
            self.pos_dict[self.timestamp] = JOINT_STATE.positions
            self.eff_dict[self.timestamp] = JOINT_STATE.efforts
            self.vel_dict[self.timestamp] = JOINT_STATE.velocities
            self.sucker_dict[self.timestamp] = [self.sucker_act_log]
            
        self.record_f.sleep()

    def sucker_act(self):
        if self.sucker_act_flag == False:
            self.sucker_act_flag = True
            self.sucker_act_log = 'on'
        else:
            self.sucker_act_flag = False
            self.sucker_act_log = 'off'

    def get_timestamp(self):
        return self.timestamp

    def read_joint_state(self):
        global JOINT_STATE
        return JOINT_STATE

    def time_callback(self, msg):
        self.current_time = msg.data
        self.secs = self.current_time.secs
        self.nsecs = self.current_time.nsecs
        self.dt_object = datetime.fromtimestamp(self.secs + self.nsecs / 1e9)
        self.formatted_time = self.dt_object.strftime('%Y-%m-%d-%H-%M-%S')


    def start_callback(self, msg):
        self.start_record_msg = msg.data

    def stop_callback(self, msg):
        if self.is_record and msg.data:
            self.write_to_excel()


    def write_to_excel(self):
        if not os.path.exists(self.root_path + self.formatted_time):
            os.mkdir(self.root_path + self.formatted_time)
        data_folder_path = os.path.join(self.root_path, self.formatted_time)
        if not os.path.exists(data_folder_path):
            os.mkdir(data_folder_path)
        writer = pd.ExcelWriter(os.path.join(data_folder_path, self.formatted_time + '.xlsx'))    
        id_1_param_pd = pd.DataFrame(self.pos_dict)
        id_2_param_pd = pd.DataFrame(self.vel_dict)
        id_3_param_pd = pd.DataFrame(self.eff_dict)
        id_4_param_pd = pd.DataFrame(self.sucker_dict)
        id_1_param_pd.to_excel(writer, sheet_name='positions', index=False, header=True)
        id_2_param_pd.to_excel(writer, sheet_name='velocities', index=False, header=True)
        id_3_param_pd.to_excel(writer, sheet_name='efforts', index=False, header=True)
        id_4_param_pd.to_excel(writer, sheet_name='sucker_actions', index=False, header=True)
        writer.close()



