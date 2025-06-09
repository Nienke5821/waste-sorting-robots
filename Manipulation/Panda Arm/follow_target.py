#!/usr/bin/env python3

import rospy
import numpy as np
from franka_msgs.msg import FrankaState
from geometry_msgs.msg import Quaternion
from tutorial_controller.msg import ImpedanceState
import tf.transformations as tft
import math
import functools
import copy
from dynamic_reconfigure.client import Client # to alter impedance gains

initial_pose = None # global variable to store the initial pose of the robot
target_pose = None # global variable to store dynamic target pose for the robot
current_pose = None # global variable to store current pose for the robot

def callbackState(msg):
    """
    Function is called every time a message about the state of the robot is received. It saves the initial pose and 
    current pose from the robot to the camera. If the initial pose is None (the first time the function is called). The
    initial pose is the current pose.

    Args:
        msg (FrankaState): ROS message type about the state of the robot.
    """

    global initial_pose, current_pose

    if initial_pose is None:

        T = np.array(msg.O_T_EE).reshape((4, 4)).transpose() 
        
        poseMsg = ImpedanceState() # convert it to a pose message as required by the low-level controller

        poseMsg.pose.position.x = T[0,3]
        poseMsg.pose.position.y = T[1,3]
        poseMsg.pose.position.z = T[2,3]
        quaternion = tft.quaternion_from_matrix(T)
        poseMsg.pose.orientation = Quaternion(*quaternion)

        initial_pose = poseMsg
        current_pose = poseMsg

def callbackTargetPose(msg):
    """
    Function called when receiving a new target pose. It saves the target pose.

    Args:
        msg (Quaternion): ROS message type about the target pose.
    """
    global target_pose
    target_pose = msg

if __name__ == '__main__':

    rospy.init_node('follow_target')

    client = Client("/panda_1/cartesian_impedance_example_controller/dynamic_reconfigure_compliance_param_node", timeout=30)
    config = client.get_configuration()                    
    config['translational_stiffness'] = 500                  
    config['rotational_stiffness'] = 30        
    client.update_configuration(config)  
    config = client.get_configuration() 
    client.close()

    rospy.Subscriber('/panda_1/franka_state_controller/franka_states', FrankaState, functools.partial(callbackState)) # subscribe to the robot state topic and target
    rospy.Subscriber('/panda_1/target_pose', ImpedanceState, callbackTargetPose)

    pub = rospy.Publisher('/panda_1/cartesian_impedance_example_controller/equilibrium_pose', ImpedanceState , queue_size=1) # publisher for transmiting the desired pose to the low-level controller

    rate = rospy.Rate(100) # set the update rate to 10 Hz
    scale_vel = 100 # scale for controller velocity
    start_time = rospy.Time.now()

    while not rospy.is_shutdown():

        if initial_pose is not None and target_pose is not None and current_pose is not None:
            elapsed_time = (rospy.Time.now() - start_time).to_sec() # calculate the time elapsed since the start

            current = current_pose.pose.position
            target = target_pose.pose.position

            direction = np.array([target.x - current.x, target.y - current.y, target.z - current.z])
            distance = np.linalg.norm(direction)

            if distance < 0.01: # when close -> keep this position
                velocity = np.array([0.0, 0.0, 0.0])
            else:
                max_step = 2  # maximum movement per cycle
                velocity = direction / distance * min(max_step, distance)

            current_pose.pose.position.x += velocity[0] # update current pose
            current_pose.pose.position.y += velocity[1]
            current_pose.pose.position.z += velocity[2] 
            current_pose.pose.orientation = copy.deepcopy(target_pose.pose.orientation) # copy target orientation
            current_pose.header.stamp = rospy.Time.now()
            current_pose.twist.linear.x = velocity[0] * scale_vel # set corresponding velocity
            current_pose.twist.linear.y = velocity[1] * scale_vel
            current_pose.twist.linear.z = velocity[2] * scale_vel

            pub.publish(current_pose) # publish current pose

        rate.sleep()
    rospy.spin()
