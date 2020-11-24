#!/usr/bin/env python3
import numpy as np
import os
import rospy
from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from duckietown_msgs.msg import Twist2DStamped, WheelEncoderStamped
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Header, Float32
from math import pi
import tf

class EncoderLocalization(DTROS):

    def __init__(self, node_name):
        """Wheel Encoder Node
        This implements basic functionality with the wheel encoders.
        """

        # Initialize the DTROS parent class
        super(EncoderLocalization, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        self.veh_name = rospy.get_namespace().strip("/")
        self.frame_id = "map"
        self.child_frame_id = "encoder_baselink"

        # Get static parameters
        self.radius = rospy.get_param(f'/{self.veh_name}/kinematics_node/radius', 0.025)
        self.L = rospy.get_param(f'/{self.veh_name}/kinematics_node/baseline', 10)
        self.rate = 30 # pose estimation frequency in hertz
        self.dt = 1/self.rate
        self.reset_vel_time = rospy.Duration(0.05)
        self.resolution = 135

        # Integrated distance
        self.encoder_ticks = [None, None] # this is updated in the callbacks
        self.encoder_ticks_prev = [None, None]
        self.encoder_tsmps = [rospy.Time.now(), rospy.Time.now()] # this is updated in the callbacks
        self.wheel_velocities = [0, 0] # this is updated in the callbacks
        self.x = 0
        self.y = 0
        self.theta = 0

        # Subscribing to the wheel encoders
        self.sub_encoder_ticks_left = rospy.Subscriber(f'/{self.veh_name}/left_wheel_encoder_node/tick', \
            WheelEncoderStamped, self.cb_encoder_data, callback_args=[0])
        self.sub_encoder_ticks_right = rospy.Subscriber(f'/{self.veh_name}/right_wheel_encoder_node/tick', \
            WheelEncoderStamped, self.cb_encoder_data, callback_args=[1])

        # Publishers - CHANGE to tf transformations rviz
        self.pub_pose = rospy.Publisher(f'/{self.veh_name}/pose', TransformStamped, queue_size=30)
        self.br = tf.TransformBroadcaster()

        # Timer
        self.reset_vel_timer = rospy.Timer(rospy.Duration(0.125), callback=self.reset_vel)
        self.timer = rospy.Timer(rospy.Duration(self.dt), callback=self.pose_callback)

        self.log("Initialized")

    def cb_encoder_data(self, msg, args):
        index_wheel = args[0]
        current_time = rospy.Time(msg.header.stamp.secs, msg.header.stamp.nsecs)
        if self.encoder_ticks[index_wheel] is not None:
            dt = current_time - self.encoder_tsmps[index_wheel]
            dN = self.encoder_ticks[index_wheel] - msg.data
            self.wheel_velocities[index_wheel] = 2*pi*self.radius*dN/(self.resolution*dt.to_sec())
        
        self.encoder_ticks[index_wheel] = msg.data
        self.encoder_tsmps[index_wheel] = current_time

    def reset_vel(self, timer_event):
        for index_wheel in [0, 1]:
            if timer_event.current_real - self.encoder_tsmps[index_wheel] > self.reset_vel_time:
                self.wheel_velocities[index_wheel] = 0.0
            
    def pose_estimation_encoder(self):
        """Does pose estimation."""
        v = -0.5*(self.wheel_velocities[1] + self.wheel_velocities[0])*self.radius
        w = -0.5*(self.wheel_velocities[1] - self.wheel_velocities[0])*self.radius/self.L

        self.x = self.x + np.cos(self.theta)*v*self.dt
        self.y = self.y + np.sin(self.theta)*v*self.dt
        self.theta = self.theta + w*self.dt

    def publish_pose_encoder(self):
        self.pose_estimation_encoder()

        msg = TransformStamped()
        q = tf.transformations.quaternion_from_euler(0, 0, self.theta)
        msg.header.frame_id = self.frame_id
        msg.child_frame_id = self.child_frame_id
        msg.transform.translation.x = self.x
        msg.transform.translation.y = self.y
        msg.transform.translation.z = 0
        msg.transform.rotation.x = q[0]
        msg.transform.rotation.y = q[1]
        msg.transform.rotation.z = q[2]
        msg.transform.rotation.w = q[3]
        if self.encoder_tsmps[0] > self.encoder_tsmps[1]:
            msg.header.stamp = self.encoder_tsmps[0]
        else:
            msg.header.stamp = self.encoder_tsmps[1]
        self.pub_pose.publish(msg)
        self.br.sendTransform((self.x, self.y, 0),
                               tf.transformations.quaternion_from_euler(0, 0, self.theta),
                               msg.header.stamp,
                               self.child_frame_id,
                               self.frame_id)
    
    def pose_callback(self, timer_event):
        self.pose_estimation_encoder()
        self.publish_pose_encoder()


if __name__ == '__main__':
    node = EncoderLocalization(node_name='encoder_localization_node')
    # Keep it spinning to keep the node alive
    rospy.spin()
    rospy.loginfo("encoder_localization_node is up and running...")