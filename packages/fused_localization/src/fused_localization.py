#!/usr/bin/env python3
import numpy as np
import os
import rospy
from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Header, Float32
from encoder_localization.srv import PoseCalibration, PoseCalibrationResponse
from math import pi
import tf

class FusedLocalization(DTROS):

    def __init__(self, node_name):
        """Wheel Encoder Node
        This implements basic functionality with the wheel encoders.
        """

        # Initialize the DTROS parent class
        super(FusedLocalization, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)

        # strings
        self.veh_name = rospy.get_namespace().strip("/")
        self.frame_id = "map"
        self.encoder_frame_id = "encoder_baselink"
        self.at_baselink_frame_id = "at_baselink"
        self.fused_baselink_frame_id = "fused_baselink"

        # Get static parameters
        self.radius = rospy.get_param(f'/{self.veh_name}/kinematics_node/radius', 0.025)
        self.L = rospy.get_param(f'/{self.veh_name}/kinematics_node/baseline', 10)
        self.rate = 30 # pose estimation frequency in hertz
        self.dt = 1/self.rate
        self.switch_duration = rospy.Duration(10.0)

        # subscribers:
        self.sub_encoder = rospy.Subscriber(f'/{self.veh_name}/pose/encoder', TransformStamped, self.encoder_callback)
        self.sub_apriltag = rospy.Subscriber(f'/{self.veh_name}/pose/apriltag', TransformStamped, self.at_callback)

        # Publishers - CHANGE to tf transformations rviz
        self.br = tf.TransformBroadcaster()
        self.transformer = tf.TransformerROS()

        # Timer
        self.timer = rospy.Timer(rospy.Duration(self.dt), callback=self.pose_callback)
        self.calibrated = False

        # poses
        self.pose_encoder = None
        self.pose_at = None
        self.tsmp_at = None
        self.pose_encoder_last = None

        self.log("Initialized")

    def encoder_callback(self, msg):
        self.pose_encoder = self.msg2pose(msg)

    def at_callback(self, msg):
        self.pose_at = self.msg2pose(msg)
        self.tsmp_at = rospy.Time(msg.header.stamp.secs, msg.header.stamp.nsecs)
        self.pose_encoder_last = self.pose_encoder # save the encoder pose at the last april tag detection

    def msg2pose(self, msg):
        x = msg.transform.translation.x
        y = msg.transform.translation.y
        z = msg.transform.translation.z
        qx = msg.transform.rotation.x
        qy = msg.transform.rotation.y
        qz = msg.transform.rotation.z
        qw = msg.transform.rotation.w
        trans = (x, y, z)
        quat = (qx, qy, qz, qw)
        return self.transformer.fromTranslationRotation(trans, quat) # pose of april tag at last detection

    def calibrate_encoder(self):
        try:
            calibrate_service = rospy.ServiceProxy(f'/{self.veh_name}/pose/calibrate_encoder', PoseCalibration) # calibration service
            trans = tf.transformations.translation_from_matrix(self.pose_at)
            quat = tf.transformations.quaternion_from_matrix(self.pose_at)
            resp = calibrate_service(trans, quat)
            self.calibrated = resp # calibrated, only done once
            if resp:
                self.log('Calibrated')
        except rospy.ServiceException as e:
            self.log("Service call failed: {}".format(e))

    def pose_callback(self, timer_event):
        pose = None
        
        if self.pose_at is not None:
            if not self.calibrated:
                self.calibrate_encoder()
                pass
            if rospy.Time.now() - self.tsmp_at > self.switch_duration:
                m_E = np.linalg.inv(self.pose_encoder_last) @ self.pose_encoder # motion of encoder since apriltag last detected
                pose = self.pose_at @ m_E

            elif rospy.Time.now() - self.tsmp_at < self.switch_duration:
                pose = self.pose_at
        if pose is None and self.pose_encoder is not None:
            pose = self.pose_encoder
        
        if pose is not None:
            t_F = tf.transformations.translation_from_matrix(pose)
            q_F = tf.transformations.quaternion_from_matrix(pose)
            self.br.sendTransform(t_F,
                                  q_F,
                                  rospy.Time.now(),
                                  self.fused_baselink_frame_id,
                                  self.frame_id)

if __name__ == '__main__':
    node = FusedLocalization(node_name='fused_localisation_node')
    # Keep it spinning to keep the node alive
    rospy.spin()
    rospy.loginfo("fused_localization_node is up and running...")

