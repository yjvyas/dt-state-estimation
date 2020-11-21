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

        # Integrated distance
        self.encoder_ticks = [None, None] # this is updated in the callbacks
        self.encoder_tsmps = [None, None] # this is updated in the callbacks
        self.wheel_velocities = [None, None] # this is updated in the callbacks
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
        self.timer = rospy.Timer(rospy.Duration(1/self.rate), callback=self.pose_estimation)

        self.log("Initialized")

    def cb_encoder_data(self, msg, args):
        index_wheel = args[0]
        if self.encoder_ticks[index_wheel] is not None:
            dN = msg.data - self.encoder_ticks[index_wheel]
            dOmega = 2*pi*self.radius*dN/msg.resolution
            dt = self.encoder_tsmps[index_wheel] - rospy.Time(msg.header.secs, msg.header.nsecs)
            self.wheel_velocities[index_wheel] = dOmega/dt.to_sec()
        self.encoder_ticks[index_wheel] = msg.data
        self.encoder_tsmps[index_wheel] = rospy.Time(msg.header.secs, msg.header.nsecs)

    def pose_estimation(self, timer_event):
        """Does pose estimation."""
        v = 0.5*(self.wheel_velocities[0] + self.wheel_velocities[1])*self.radius
        omega = 0.5*(self.wheel_velocities[1] - self.wheel_velocities[0])*self.radius/self.baseline
        dt = timer_event.current_real - timer_event.last_real
        self.x = self.x + np.cos(self.theta)*v*dt
        self.y = self.y + np.sin(self.theta)*v*dt
        self.theta = self.theta + omega*dt

        msg = TransformStamped()
        q = tf.transform.quaternion_about_axis(self.theta, (0,0,1))
        msg.header.frame_id = self.frame_id
        msg.header.child_frame_id = self.child_frame_id
        msg.transform.translation.x = self.x
        msg.transform.translation.y = self.y
        msg.transform.translation.z = 0
        msg.transform.rotation.x = q[0]
        msg.transform.rotation.y = q[1]
        msg.transform.rotation.z = q[2]
        msg.transform.rotation.w = q[3]
        if self.encoder_tsmps[0] > self.encoder_tsmps[1]:
            msg.stamp = self.encoder_tsmps[0]
        else:
            msg.stamp = self.encoder_tsmps[1]
        self.pub_pose.publish(msg)
        self.br.sendTransform(msg)


if __name__ == '__main__':
    node = EncoderLocalization(node_name='wheel_encoder_odometry')
    # Keep it spinning to keep the node alive
    rospy.spin()
    rospy.loginfo("wheel_encoder_node is up and running...")