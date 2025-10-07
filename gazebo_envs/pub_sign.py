import rospy
from std_msgs.msg import Bool, Time
from datetime import datetime
import subprocess
from joint_states_process import JointStateRead
from gen_rand_redbox import GenRandBox
import time
from take_rand_box import take_rand_box
from gen_rand_object import GenRandObj
from publisher.go_home import go_home
import random
from move_test import move_to


'''
collect datasets main code
'''


class RecordDataControl:
    def __init__(self):
        self.start_pub = rospy.Publisher('/recorddatasets/start', Bool, queue_size=10)
        self.stop_pub = rospy.Publisher('/recorddatasets/stop', Bool, queue_size=10)


    def send_start_signal(self):
        rospy.loginfo("Sending start signal")
        self.start_pub.publish(Bool(data=True))
        self.stop_pub.publish(Bool(data=False))

    def send_stop_signal(self):
        rospy.loginfo("Sending stop signal")
        self.stop_pub.publish(Bool(data=True))
        self.start_pub.publish(Bool(data=False))

class GenBoxPubSignal:
    def __init__(self):
        self.random_box = rospy.Publisher('/spawn_signal', Bool, queue_size=10)  # signal publisher

    def send_gen_signal(self):
        rospy.loginfo("Sending generation signal")
        self.random_box.publish(Bool(data=True))

    def send_remain_signal(self):
        rospy.loginfo("Sending remaining signal")
        self.random_box.publish(Bool(data=False))


class TimePublisher:
    def __init__(self):
        self.time_pub = rospy.Publisher('/timestamp', Time, queue_size=10)
        self.current_time = ''

    def publish_time(self):
        self.current_time = rospy.Time.now()
        self.time_pub.publish(self.current_time)
        rospy.loginfo("Published time: %s", self.current_time)

    def time_process(self):
        if self.current_time == '':
            return
        self.secs = self.current_time.secs
        self.nsecs = self.current_time.nsecs
        self.dt_object = datetime.fromtimestamp(self.secs + self.nsecs / 1e9)
        self.formatted_time = self.dt_object.strftime('%Y-%m-%d-%H-%M-%S')

    def get_now_time(self):
        self.time_process()
        return self.formatted_time

class RunCamProcessPy2:
    def __init__(self):
        python2_cam_process = './cam_process.py'
        self.process = subprocess.Popen([python2_cam_process], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def stop(self):
        self.process.terminate()
        self.process.wait()


from gazebo_msgs.srv import DeleteModel

class DeleteGazeboModel:
    def __init__(self):
        try:
            rospy.init_node('delete_gazebo_model', anonymous=True)
        except:
            pass

    def delete_model(self, model_name):
        rospy.wait_for_service('/gazebo/delete_model')
        try:
            delete_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
            resp = delete_model(model_name)
            if resp.success:
                rospy.loginfo(f"Deleted model: {model_name}")
            else:
                rospy.logerr(f"Failed to delete model: {model_name}")
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")


def save_info(tar_x, tar_y, tar_z, tar_r, gripper_pos_offset_x, 
              gripper_pos_offset_y, gripper_pos_offset_z, gripper_pos_offset_r,
              pd_pos_x, pd_pos_y, pd_pos_z,
              color, model_name, current_time, is_fixed_tar_pos:bool=True,
              is_fixed_gen_pos:bool=True, is_sucessed:bool=True):

    info = f'''
    obj_pos_x:{tar_x:.4f}
    obj_pos_y:{tar_y:.4f}
    obj_pos_z:{tar_z:.4f}
    obj_pos_r:{tar_r:.4f}
    gripper_pos_offset_x:{gripper_pos_offset_x:.4f}
    gripper_pos_offset_y:{gripper_pos_offset_y:.4f}
    gripper_pos_offset_z:{gripper_pos_offset_z:.4f}
    gripper_pos_offset_r:{gripper_pos_offset_r:.4f}
    putdown_pos_x:{pd_pos_x}
    putdown_pos_y:{pd_pos_y}
    putdown_pos_z:{pd_pos_z}
    obj_color:{color}
    model_name:{model_name}
    is_fixed_tar_pos:{is_fixed_tar_pos}
    is_fixed_gen_pos:{is_fixed_gen_pos}
    is_sucessed:{is_sucessed}
    other:
    '''

    with open(f'./datasets/{current_time}/info.txt', 'w+') as f:
        f.write(info)



if __name__ == '__main__':
    rospy.init_node('record_datasets_control', anonymous=True)
    is_manual = False
    command_gen = ''
    control = RecordDataControl()
    time_publisher = TimePublisher()
    cam_process = RunCamProcessPy2()
    joint_read = JointStateRead()
    gen_box_signal = GenBoxPubSignal()
    gen_box = GenRandBox()
    deleter = DeleteGazeboModel()
    time.sleep(1)
    go_home()
    time.sleep(1)
    try:
        deleter.delete_model('red_box')
    except rospy.ROSInterruptException as e:
        print(e)
    # gen_obj = GenRandObj()
    

    # delete model

    try:
        while not rospy.is_shutdown():
            go_home()
            time.sleep(1)
            # start record
            if is_manual:
                command_gen = input("Enter 'y' to begin recording:").strip().lower()
            if command_gen == 'y' or not is_manual:
                time_publisher.publish_time()
                control.send_start_signal()
            # generate object
            if is_manual:
                command_gen = input("Enter 'y' to begin generate a red box or 'n' to remain unchange: ").strip().lower()
            if command_gen == 'y' or not is_manual:
                gen_box_signal.send_gen_signal()
                time.sleep(1)
                tar_x, tar_y, tar_z, tar_r, color, model_name, gripper_pos_offset_x, \
                    gripper_pos_offset_y, gripper_pos_offset_z, gripper_pos_offset_r \
                    = gen_box.get_ran_box_pos()
            elif command_gen == 'n' or not is_manual:
                gen_box_signal.send_remain_signal()
            else:
                print("Invalid command. Please enter 'y' or 'n'.")
            time.sleep(1)
            # pickup 
            if is_manual:
                command_gen = input("Enter 'y' to take red box or 'n' to remain unchange: ").strip().lower()
            if command_gen == 'y' or not is_manual:
                # z fixed to 0.01
                pd_pos_x = random.randint(-170, 70)
                pd_pos_y = random.randint(-200, -70)

                sucker_on_t, sucker_off_t, final_pos = take_rand_box(x=tar_x, y=tar_y, z=tar_z, r=tar_r, pd_pos_x=pd_pos_x, pd_pos_y=pd_pos_y, joint_read=joint_read)
                # sucker_on_t, sucker_off_t = take_rand_box(x=tar_x, y=tar_y, z=tar_z, r=tar_r, joint_read=joint_read)

            elif command_gen == 'n':
                pass
            else:
                print("Invalid command. Please enter 'y' or 'n'.")
            time.sleep(1)
            # stop record 
            if is_manual:
                command_gen = input("Enter 'y' to stop recording or 'n' to end recording: or 's' to skip this part:").strip().lower()
            if command_gen == 'y' or not is_manual:
                control.send_stop_signal()
                time.sleep(1)
                move_to([[0, -150, 150]], final_pos=final_pos)
            time.sleep(1)
            # delete object models
            if is_manual:
                command_gen = input("Enter 'y' to delete box or 'n' to remain unchange: ").strip().lower()
            if command_gen == 'y' or not is_manual:
                try:
                    deleter.delete_model(model_name)
                except rospy.ROSInterruptException as e:
                    print(e)
            time.sleep(1)
            save_info(tar_x, tar_y, tar_z, tar_r, gripper_pos_offset_x, 
              gripper_pos_offset_y, gripper_pos_offset_z, gripper_pos_offset_r, 
              pd_pos_x, pd_pos_y, pd_pos_z=61, color=color, model_name=model_name, 
              current_time=time_publisher.get_now_time(), is_sucessed=True)
            time.sleep(3)
    except rospy.ROSInterruptException:
        save_info(tar_x, tar_y, tar_z, tar_r, gripper_pos_offset_x, 
              gripper_pos_offset_y, gripper_pos_offset_z, gripper_pos_offset_r, 
              pd_pos_x, pd_pos_y, pd_pos_z=61, color=color, model_name=model_name, 
              current_time=time_publisher.get_now_time(),is_sucessed=False)
    finally:
        cam_process.stop()
