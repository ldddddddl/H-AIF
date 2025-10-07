#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool
from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Pose
import random
import os
import subprocess

OBJECT_PATH = "/home/ubuntu/jetmax/src/jetmax_description/urdf/objects"

class GenRandObj:
    def __init__(self):
        try:
            rospy.init_node('obj_spawner', anonymous=True)
            rospy.loginfo("ROS node initialized.")
        except rospy.ROSException as e:
            # rospy.logerr(f"ROS initialization failed: {e}")
            pass
            # return
        
        rospy.Subscriber('/spawn_signal', Bool, self.callback)
        rospy.loginfo("Subscribed to /spawn_signal.")
        self.obj_count = 0
        self.model_name = ''
        self.tart_r = -1
        self.tart_x = 0.0
        self.tart_y = 0.0

    def spawn_model(self, model_name='', model_type=None):
        folders = os.listdir(OBJECT_PATH)
        if not folders:
            rospy.logerr("No folders found in OBJECT_PATH.")
            return
        
        model_name = random.choice(folders)
        try:
            _, self.color = model_name.split("_")
        except:
            self.color = ''
        rospy.loginfo(f"Selected folder: {model_name}")

        objs = os.listdir(os.path.join(OBJECT_PATH, model_name))
        model_type = None
        for f in objs:
            if f.endswith('.sdf'):
                model_type = 'sdf'
                file_path = os.path.join(OBJECT_PATH, model_name, f)
                break
            elif f.endswith('.xacro'):
                model_type = 'xacro'
                file_path = os.path.join(OBJECT_PATH, model_name, f)
                break
        
        if model_type is None:
            rospy.logerr(f"No sdf or xacro file found in {model_name}.")
            return

        if model_type == "xacro":
            # Convert xacro file to urdf
            urdf_file_path = os.path.join(OBJECT_PATH, model_name, 'model.urdf')
            try:
                subprocess.check_call(['rosrun', 'xacro', 'xacro', file_path, '-o', urdf_file_path])
                file_path = urdf_file_path
                rospy.loginfo(f"Converted xacro to urdf: {file_path}")
            except subprocess.CalledProcessError as e:
                rospy.logerr(f"xacro to urdf conversion failed: {e}")
                return
        
        try:
            with open(file_path, 'r') as f:
                model_xml = f.read()
            rospy.loginfo(f"Read model file: {file_path}")
        except IOError as e:
            rospy.logerr(f"Failed to open model file {file_path}: {e}")
            return 

        # Define pose of the model
        self.pose = Pose()
        self.pose.position.x = random.uniform(0.05, 0.2)
        self.pose.position.y = random.uniform(-0.2, -0.05)
        self.tart_x = self.pose.position.x
        self.tart_y = self.pose.position.y 
        self.pose.position.z = 0.0
        rospy.loginfo(f"Generated pose: x={self.pose.position.x}, y={self.pose.position.y}, z={self.pose.position.z}")

        # Determine which service to use for model spawning
        if model_type == "sdf":
            service_name = '/gazebo/spawn_sdf_model'
        elif model_type == "xacro":
            service_name = '/gazebo/spawn_urdf_model'
        else:
            rospy.logerr("Unsupported model type.")
            return

        # Call service to spawn model
        rospy.wait_for_service(service_name)
        try:
            spawn_model = rospy.ServiceProxy(service_name, SpawnModel)
            self.obj_count += 1
            self.model_name = model_name+str(self.obj_count)
            resp = spawn_model(self.model_name, model_xml, '', self.pose, 'world')
            rospy.loginfo(f"Spawned new model {model_name} at {self.pose.position.x}, {self.pose.position.y}, {self.pose.position.z}")
        except rospy.ServiceException as e:
            rospy.logerr(f"Spawn model service call failed: {e}")
        
    def callback(self, msg):
        if msg.data:
            rospy.loginfo("Received signal to spawn model.")
            self.spawn_model()

    def get_ran_box_pos(self):
        return [self.tart_x * 887.3, self.tart_y * 901.6, 85, self.tart_r, self.color, self.model_name]



# if __name__ == '__main__':
#     GenRandObj()
#     rospy.spin()
