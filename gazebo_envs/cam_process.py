#!/usr/bin/env python2.7
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge, CvBridgeError
import cv2
import time
from datetime import datetime
from std_msgs.msg import Time
import os


class VideoRecorder:
    def __init__(self, subsc_path, test_flag=False):
        '''
        '/rrbot/side_cam/image_raw'
        '/rrbot/gripper_cam/image_raw'
        '''
        self.test_flag = test_flag
        self.bridge = CvBridge()
        self.recording = False
        self.subsc_path = subsc_path
        self.image_sub = rospy.Subscriber('/rrbot/' + subsc_path + '/image_raw', Image, self.image_callback)
        self.time_sub = rospy.Subscriber('/timestamp', Time, self.time_callback)
        self.start_sub = rospy.Subscriber('/recorddatasets/start', Bool, self.start_callback)
        self.stop_sub = rospy.Subscriber('/recorddatasets/stop', Bool, self.stop_callback)
        self.video_writer = None
        self.formatted_time = ''


    def time_callback(self, msg):
        self.current_time = msg.data
        self.secs = self.current_time.secs
        self.nsecs = self.current_time.nsecs
        self.dt_object = datetime.fromtimestamp(self.secs + self.nsecs / 1e9)
        self.formatted_time = self.dt_object.strftime('%Y-%m-%d-%H-%M-%S')

    def start_callback(self, msg):
        if msg.data and not self.recording:
            # rospy.loginfo("pre_start, startmsgdata:{}".format(msg.data))
            if not os.path.exists('./datasets/' + self.formatted_time):
                os.mkdir('./datasets/' + self.formatted_time)
            self.recording = True
            self.video_writer = cv2.VideoWriter('./datasets/{}/{}_{}.avi'.format(self.formatted_time, self.subsc_path, 
                                                                                 self.formatted_time), cv2.VideoWriter_fourcc(*'XVID'), 24.0, (640, 480))
            rospy.loginfo("Video recording started.")

    def stop_callback(self, msg):

        if msg.data and self.recording:
            # rospy.loginfo("pre_stop, stopmsgdata:{}".format(msg))

            self.recording = False
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            rospy.loginfo("Video recording stopped.")

    def image_callback(self, data):
        if self.recording and self.video_writer is not None:
            try:
                cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
                self.video_writer.write(cv_image)
            except CvBridgeError as e:
                rospy.logerr(e)

    def shutdown(self):
        if self.video_writer is not None:
            self.video_writer.release()
        cv2.destroyAllWindows()

    def get_current_time(self):
        return self.formatted_time


if __name__ == '__main__':
# def save_video():
    try:
        rospy.init_node('video_recorder', anonymous=True)
    except:
        pass
    gripper_video_recorder = VideoRecorder(subsc_path='gripper_cam')
    side_video_recorder = VideoRecorder(subsc_path='side_cam')
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")
    finally:
        gripper_video_recorder.shutdown()
        side_video_recorder.shutdown()
