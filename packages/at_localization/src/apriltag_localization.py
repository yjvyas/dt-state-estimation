#!/usr/bin/env python3
import numpy as np
import os
import math
import cv2

import rospy
import yaml
import sys
import tf
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TransformStamped
from dt_apriltags import Detector

"""

This is a template that can be used as a starting point for the CRA1 exercise.
You need to project the model file in the 'models' directory on an AprilTag.
To help you with that, we have provided you with the Renderer class that render the obj file.

"""

class AprilTag_Localization(DTROS):

    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(AprilTag_Localization, self).__init__(node_name=node_name,node_type=NodeType.GENERIC)
        self.veh_name = rospy.get_namespace().strip("/")

        # read calibration

        intrinsic_fname = '/data/config/calibrations/camera_intrinsic/default.yaml'
        if 'INTRINSICS_FILE' in os.environ:
            intrinsic_fname = os.environ['CALIBRATION_FILE']
        intrinsics = self.readYamlFile(intrinsic_fname)
        self.D = np.array(intrinsics['distortion_coefficients']['data'])
        self.K = np.reshape(np.array(intrinsics['camera_matrix']['data']), [3,3])

        # init poses

        self.x = 0
        self.y = 0
        self.theta = 0

        # transforms
        self.A_to_map = tf.transformations.identity_matrix() # april-tag
        self.A_to_map[2,3] = -0.095

        self.Ad_to_A = np.array([[0, 1, 0 , 0], [0, 0, -1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
        self.C_to_Cd = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        self.baselink_to_camera = tf.transformations.rotation_matrix(np.pi*(20/180), (0, 1, 0))
        self.baselink_to_camera[0:3,3] = np.array([0.0582, 0, 0.1072])

        # april tag detector
        self.at_detector = Detector(searchpath=['apriltags'],
                                    families='tag36h11',
                                    nthreads=4,
                                    quad_decimate=2.0,
                                    quad_sigma=0.0,
                                    refine_edges=1,
                                    decode_sharpening=0.25,
                                    debug=0)


        # subscriber for images
        self.listener = tf.TransformListener()
        self.sub = rospy.Subscriber(f'/{self.veh_name}/camera_node/image/compressed', CompressedImage, self.image_callback)

        # publisher
        self.pub_pose = rospy.Publisher(f'/{self.veh_name}/pose/apriltag', TransformStamped, queue_size=30)
        self.br = tf.TransformBroadcaster()

        # pther important things
        self.cvbr = CvBridge()


    def image_callback(self, msg):
        """Detects april-tags and renders duckie on them."""
        img = self.readImage(msg)
    
        stamp = rospy.Time(msg.header.stamp.secs, msg.header.stamp.nsecs)
        if len(img) > 0:
            K = self.K
            tags = self.at_detector.detect(img[:,:,0], 
                                           estimate_tag_pose=True, 
                                           camera_params=[K[0,0], K[1,1], K[0,2], K[1,2]],
                                           tag_size=0.0675)

            ### TODO: Code here to calculate pose of camera (?)

            self.broadcast_pose(self.baselink_to_camera, 'baselink', 'camera', stamp)
            if len(tags) > 0:
                at_camera = tf.transformations.identity_matrix()
                at_camera[0:3,0:3] = np.array(tags[0].pose_R) ## only detects first tag!
                at_camera[0:3,3] = np.array(tags[0].pose_t).flatten()
                at = self.C_to_Cd @ at_camera @ self.Ad_to_A
                self.broadcast_pose(at, 'camera', 'apriltag', stamp)
                self.broadcast_pose(self.A_to_map, 'apriltag', 'map', stamp)

                try:
                    (trans, q) = self.listener.lookupTransform('/baselink', '/map', rospy.Time(0))
                    msg = TransformStamped()
                    msg.header.frame_id = 'map'
                    msg.child_frame_id = 'baselink'
                    msg.transform.translation.x = trans[0]
                    msg.transform.translation.y = trans[1]
                    msg.transform.translation.z = trans[2]
                    msg.transform.rotation.x = q[0]
                    msg.transform.rotation.y = q[1]
                    msg.transform.rotation.z = q[2]
                    msg.transform.rotation.w = q[3]
                    msg.header.stamp = stamp
                    self.pub_pose.publish(msg)
                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    return


    def broadcast_pose(self, pose, frame_id, child_frame_id, stamp):
        q = tf.transformations.quaternion_from_matrix(pose)
        self.br.sendTransform((pose[0,3], pose[1,3], pose[2,3]),
                               q,
                               stamp,
                               child_frame_id,
                               frame_id)

    def readImage(self, msg_image):
        """
            Convert images to OpenCV images
            Args:
                msg_image (:obj:`CompressedImage`) the image from the camera node
            Returns:
                OpenCV image
        """
        try:
            cv_image = self.cvbr.compressed_imgmsg_to_cv2(msg_image)
            image_undistorted = cv2.undistort(cv_image, self.K, self.D) # TODO: move this to GPU.
            return image_undistorted
        except CvBridgeError as e:
            self.log(e)
            return []
        except AttributeError as e:
            return []

    def readYamlFile(self,fname):
        """
            Reads the 'fname' yaml file and returns a dictionary with its input.

            You will find the calibration files you need in:
            `/data/config/calibrations/`
        """
        with open(fname, 'r') as in_file:
            try:
                yaml_dict = yaml.load(in_file)
                return yaml_dict
            except yaml.YAMLError as exc:
                self.log("YAML syntax error. File: %s fname. Exc: %s"
                         %(fname, exc), type='fatal')
                rospy.signal_shutdown()
                return


    def onShutdown(self):
        super(AprilTag_Localization, self).onShutdown()


if __name__ == '__main__':
    # Initialize the node
    camera_node = AprilTag_Localization(node_name='apriltag_localization_node')
    # Keep it spinning to keep the node alive
    rospy.spin()