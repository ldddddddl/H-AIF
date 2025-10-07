import rospy
from std_msgs.msg import Bool
from gazebo_msgs.srv import SpawnModel
from geometry_msgs.msg import Pose
import random

BOX_COUNT = 0

class GenRandBox:

    def __init__(self):
        try:
            rospy.init_node('box_spawner', anonymous=True)
        except:
            pass
        rospy.Subscriber('/spawn_signal', Bool, self.callback)
        self.tart_x = 0.0
        self.tart_y = 0.0
        self.tart_r = 0.02
        self.color = 'Red'
        self.model_name = ''
        self.gripper_pos_offset_x, self.gripper_pos_offset_y, self.gripper_pos_offset_z, \
            self.gripper_pos_offset_r = 0.0, 0.0, 0.0, 0.0
        
        self.colors = ["Blue", "Yellow", "Green", "Red", "Orange", "Purple", "SkyBlue", 
                    "RedBright","ZincYellow","DarkYellow","Turquoise","Indigo",
                    "RedGlow","GreenGlow","BlueGlow","YellowGlow","PurpleGlow","TurquoiseGlow",
                    "TurquoiseGlowOutline","WoodFloor","CeilingTiled","PaintedWall","Gold","WoodPallet",
                    "Wood","Bricks","Road","Residential","Tertiary","Steps"]

    def spawn_box(self):
        global BOX_COUNT
        self.rand_r = random.randint(150, 480) / 10000
        # self.rand_r = 0.05
        self.color = random.choice(self.colors)
        mass_center = self.rand_r / 2
        collision_center = self.rand_r * 2
        self.tart_r = mass_center + self.rand_r
        model_xml = f"""
        <?xml version="1.0"?>
        <robot name="random_box" xmlns:xacro="http://ros.org/wiki/xacro">
        <link name="v_random_box">
            <collision>
                <origin xyz="0 0 0" rpy="0 0 0" />
                <geometry>
                    <box size="{collision_center} {collision_center} {collision_center}" />
                </geometry>
            </collision>
            <visual>
                <origin xyz="0 0 {0 - mass_center}" rpy="0 0 0" />
                <geometry>
                    <box size="{self.rand_r} {self.rand_r} {self.rand_r}" />
                </geometry>
            </visual>
            <inertial>
                <origin xyz="0 0 {mass_center}" rpy="0 0 0" />
                <mass value="0.1" />
            <inertia ixx="0.0000033333" ixy="0.0" ixz="0.0" iyy="0.0000033333" iyz="0.0" izz="0.0000033333" />
            </inertial>
        </link>
        <gazebo reference="v_random_box">
            <material>Gazebo/{self.color}</material>
            <mu1>3</mu1>
            <mu2>3</mu2>
        </gazebo>
        </robot>
        """
        
        self.tart_x = random.uniform(0.07, 0.2)
        self.tart_y = random.uniform(-0.2, -0.07)
        # self.tart_x = 0.17
        # self.tart_y = -0.05
        self.pose = Pose()
        self.pose.position.x = self.tart_x
        self.pose.position.y = self.tart_y
        self.pose.position.z = 0.0


        rospy.wait_for_service('/gazebo/spawn_urdf_model')
        try:
            spawn_model = rospy.ServiceProxy('/gazebo/spawn_urdf_model', SpawnModel)
            self.model_name = f'random_box{BOX_COUNT}'
            spawn_model(self.model_name, model_xml, '', self.pose, 'world')
            BOX_COUNT += 1
            # rospy.loginfo(f'Spawned new box at {self.pose.position.x}, {self.pose.position.y}, {self.pose.position.z}')
            rospy.loginfo(f'Spawned new box at {self.tart_x}, {self.tart_y}, {self.pose.position.z}, {self.rand_r}')
        except rospy.ServiceException as e:
            rospy.logerr(f"Spawn model service call failed: {e}")

    def callback(self, msg):

        if msg.data:
            self.spawn_box()

    def get_ran_box_pos(self):
        return [self.tart_x * 900.3, self.tart_y * 910.6, 61, self.rand_r * 1000, self.color, self.model_name,
                self.gripper_pos_offset_x, self.gripper_pos_offset_y, self.gripper_pos_offset_z, self.gripper_pos_offset_r,]

if __name__ == "__main__":
    gen = GenRandBox()
    for c in gen.colors:
        if input() == '':
            print(c)
            gen.spawn_box(c)
        else:
            break