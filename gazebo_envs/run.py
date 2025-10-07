import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from joint_states_process import JointStateRead
from options import read_config
from model.models import CausalNet, ActNet
from model.model_utils import initialize_parameters_xavier as init_param
from dataset_convert import perc_trans
from losses import perc_loss
from misc import random_target, TensorContainer
import sys
sys.path.append(r"/home/ubuntu/Desktop/action_generation/publisher")
# from publisher import publisher
from publisher.go_home import go_home
from publisher.act_publisher import joint_publisher, publisher_callback
import rospy
import time
from jmdataload import create_dataloader


ParamConfig = read_config()

def run():
    is_use_cuda = torch.cuda.is_available()
    model = ActNet()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=ParamConfig.lr)

    if ParamConfig.is_init_param:    
        init_param(model)

    joint_publisher()
    go_home()
    joint = JointStateRead()
    rate = rospy.Rate(0.5)
    iter_count = 0
    while True:
        iter_count += 1
        # node_joint_state_listener()
        joint_data_obj = joint.read_joint_state()
        # print(joint_data_obj)
        percetions = perc_trans(joint_data_obj)
        model.train()
        joint_results = model(percetions)
        random_targets = random_target(joint_data_obj)
        losses = perc_loss(joint_results, random_targets, ParamConfig)
        optimizer.zero_grad()   
        losses.backward()
        optimizer.step()
        action = random_targets.pos[:, :-1].squeeze(0).tolist()
        # node_ik_jetmax()
        # rospy.loginfo(f"perc_pos:{joint_data_obj}")
        rospy.loginfo(f"iter_count:{iter_count}")
        publisher_callback(action)
        rate.sleep()
        go_home()
        rate.sleep()
    # print(pos_with_time_tensor)
    


def node_joint_state_listener():
    try:
        # Ensure node1 is shut down
        rospy.signal_shutdown("Switching to joint_state_listener")
        time.sleep(1)  # Wait for node1 to shutdown (optional) 
    except rospy.ROSInterruptException:
        pass

    rospy.init_node('joint_state_listener', anonymous=True)
    rospy.loginfo("Node 'joint_state_listener' started")
    rospy.spin()

def node_ik_jetmax():
    try:
        # Ensure node1 is shut down
        rospy.signal_shutdown("Switching to ik_jetmax")
        time.sleep(1)  # Wait for node1 to shutdown (optional)
    except rospy.ROSInterruptException:
        pass
    rospy.init_node('ik_jetmax', log_level=rospy.DEBUG)
    rospy.loginfo("Node 'ik_jetmax' started")
    rospy.spin()




if __name__ == "__main__":
    if not rospy.core.is_initialized():
        rospy.init_node('act_generate', anonymous=True, log_level=rospy.DEBUG)
    if ParamConfig.state == "training":
        main()
    else:
        run()









