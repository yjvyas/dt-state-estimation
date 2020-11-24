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
        self.frame_id = "map"

        # read calibration
        calibration_fname = '/data/config/calibrations/camera_extrinsic/default.yaml'
        if 'CALIBRATION_FILE' in os.environ:
            calibration_fname = os.environ['CALIBRATION_FILE']
        
        self.homography = self.readYamlFile(calibration_fname)

        intrinsic_fname = '/data/config/calibrations/camera_intrinsic/default.yaml'
        if 'INTRINSICS_FILE' in os.environ:
            intrinsic_fname = os.environ['CALIBRATION_FILE']
        intrinsics = self.readYamlFile(intrinsic_fname)
        self.D = np.array(intrinsics['distortion_coefficients']['data'])
        self.K = np.reshape(np.array(intrinsics['camera_matrix']['data']), [3,3])

        H = np.reshape(np.array(self.homography['homography']), [3, 3])
        self.camera_to_axle = np.dot(np.array([[1, 0, 0, -0.0582], [0, 1, 0, 0], [0, 0, 1, -0.1042], [0, 0, 0, 1]]), 
                                     tf.transformations.rotation_matrix(np.pi/6, (0, 1, 0)))

        # init poses

        self.x = 0
        self.y = 0
        self.theta = 0

        # transforms
        self.at_map = tf.transformations.identity_matrix() # april-tag
        self.at_map[2,3] = 0.095
        
        self.C_to_Cd = np.array([[0, -1, 0 , 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])

        self.Ad_to_a = np.array([[0, 0, -1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])

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
        self.sub = rospy.Subscriber(f'/{self.veh_name}/camera_node/image/compressed', CompressedImage, self.image_callback)

        # publisher
        self.pub_pose = rospy.Publisher(f'/{self.veh_name}/pose', TransformStamped, queue_size=30)
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

            self.broadcast_pose(self.at_map, self.frame_id, 'apriltag', stamp)

            if len(tags) > 0:            
                at_camera = tf.transformations.identity_matrix()
                at_camera[0:3,0:3] = np.array(tags[0].pose_R) ## only detects first tag!
                at_camera[0:3,3] = np.array(tags[0].pose_t).flatten()
                camera_map = self.C_to_Cd @ at_camera @  self.Ad_to_a
                baselink = camera_map @ self.camera_to_axle 
                self.broadcast_pose(camera_map, self.frame_id, 'camera', stamp)
                msg_baselink = self.broadcast_pose(baselink, self.frame_id, 'at_baselink', stamp)
                self.pub_pose.publish(msg_baselink)


    def broadcast_pose(self, pose, frame_id, child_frame_id, stamp):
        msg = TransformStamped()
        q = tf.transformations.quaternion_from_matrix(pose)
        msg.header.frame_id = frame_id
        msg.child_frame_id = child_frame_id
        msg.transform.translation.x = pose[0,3]
        msg.transform.translation.y = pose[1,3]
        msg.transform.translation.z = pose[2,3]
        msg.transform.rotation.x = q[0]
        msg.transform.rotation.y = q[1]
        msg.transform.rotation.z = q[2]
        msg.transform.rotation.w = q[3]
        msg.header.stamp = stamp
        self.br.sendTransform((pose[0,3], pose[1,3], pose[2,3]),
                               q,
                               stamp,
                               child_frame_id,
                               self.frame_id)
        return msg

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