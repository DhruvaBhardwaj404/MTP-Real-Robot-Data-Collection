#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import time
import os
import cv2  # Added for resizing
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge, CvBridgeError
from std_srvs.srv import SetBool, SetBoolResponse, Trigger, TriggerResponse

class ControlledDataCollector:
    def __init__(self):
        rospy.init_node('franka_realsense_data_collector', anonymous=True)

        self.is_recording = False
        self._discard_flag = False
        self.bridge = CvBridge()

        # Target resolution
        self.target_width = 640
        self.target_height = 480
        rospy.loginfo(f"Downsizing images to {self.target_width}x{self.target_height}")

        # --- Latest sensor snapshots ---
        self.latest_image_data_1 = None
        self.latest_image_data_2 = None
        self.latest_joint_positions = None
        self.latest_gripper_positions = None

        # --- Binary gripper state tracking ---
        # Start with a default state (1 = opening) so recording isn't blocked
        self.gripper_binary_state = 1
        self.previous_gripper_width = None       

        # --- Parameters ---
        self.gripper_deadband_m = rospy.get_param('~gripper_deadband_m', 0.0005)
        self.save_dir = rospy.get_param('~save_dir', '.')
        # Using specific topics for EIH (Eye-In-Hand) and External cameras
        self.cam1_topic = rospy.get_param('~cam1_topic', '/eih/color/image_raw')
        self.cam2_topic = rospy.get_param('~cam2_topic', '/ext/color/image_raw')

        # --- Data buffer ---
        self.collected_data = []

        # --- Subscribers ---
        self.image_sub_1 = rospy.Subscriber(self.cam1_topic, Image, self.image_callback_1, queue_size=1)
        self.image_sub_2 = rospy.Subscriber(self.cam2_topic, Image, self.image_callback_2, queue_size=1)
        self.joint_sub = rospy.Subscriber("/franka_state_controller/joint_states", JointState, self.joint_callback, queue_size=1)
        self.gripper_joint_sub = rospy.Subscriber("/franka_gripper/joint_states", JointState, self.gripper_joint_callback, queue_size=1)

        # --- Services ---
        self.control_service = rospy.Service('/data_collector/set_recording', SetBool, self.handle_set_recording)
        self.discard_service = rospy.Service('/data_collector/discard', Trigger, self.handle_discard)

        # --- Collection timer: 10 Hz ---
        self.collection_rate = 10.0
        rospy.Timer(rospy.Duration(1.0 / self.collection_rate), self.timer_callback)

        rospy.loginfo("Dual-Camera Data Collector READY (Immediate Start Mode).")

    # ------------------------------------------------------------------ #
    #  Subscriber Callbacks with Downsizing                            #
    # ------------------------------------------------------------------ #

    def image_callback_1(self, msg):
        try:
            # Convert ROS message to OpenCV BGR image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # UPDATED: Resize image to target resolution
            if cv_image is not None and cv_image.size > 0:
                resized_image = cv2.resize(cv_image, (self.target_width, self.target_height))
                self.latest_image_data_1 = np.array(resized_image, dtype=np.uint8)
            else:
                rospy.logwarn_throttle(2, "Empty or invalid image from cam1, skipping resize.")
        except CvBridgeError as e:
            rospy.logerr(f"Cam1 image processing error: {e}")

    def image_callback_2(self, msg):
        try:
            # Convert ROS message to OpenCV BGR image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # UPDATED: Resize image to target resolution
            if cv_image is not None and cv_image.size > 0:
                resized_image = cv2.resize(cv_image, (self.target_width, self.target_height))
                self.latest_image_data_2 = np.array(resized_image, dtype=np.uint8)
            else:
                rospy.logwarn_throttle(2, "Empty or invalid image from cam2, skipping resize.")
        except CvBridgeError as e:
            rospy.logerr(f"Cam2 image processing error: {e}")

    # ------------------------------------------------------------------ #
    #  Joint and Gripper Callbacks (Unchanged)                         #
    # ------------------------------------------------------------------ #

    def joint_callback(self, msg):
        if len(msg.position) >= 7:
            self.latest_joint_positions = np.array(msg.position[:7], dtype=np.float64)

    def gripper_joint_callback(self, msg):
        if len(msg.position) >= 2:
            new_positions = np.array(msg.position[:2], dtype=np.float64)
            self.latest_gripper_positions = new_positions
            self._update_binary_gripper_state(new_positions)

    def _update_binary_gripper_state(self, new_positions):
        current_width = float(new_positions[0] + new_positions[1])
        if self.previous_gripper_width is not None:
            delta = current_width - self.previous_gripper_width
            if delta < -self.gripper_deadband_m:
                self.gripper_binary_state = 0   # closing
            elif delta > self.gripper_deadband_m:
                self.gripper_binary_state = 1   # opening
        self.previous_gripper_width = current_width

    # ------------------------------------------------------------------ #
    #  Timer — Core Collection Loop                                    #
    # ------------------------------------------------------------------ #

    def timer_callback(self, event):
        if not self.is_recording or self._discard_flag:
            if self._discard_flag: self._do_discard()
            return

        # Check for mandatory data
        if any(v is None for v in [self.latest_image_data_1, self.latest_image_data_2, 
                                   self.latest_joint_positions, self.latest_gripper_positions]):
            rospy.logwarn_throttle(5, "Waiting for all sensor topics...")
            return

        # Recording logic, including the current gripper state
        self.collected_data.append((
            np.copy(self.latest_image_data_1),
            np.copy(self.latest_image_data_2),
            np.copy(self.latest_joint_positions),
            np.copy(self.latest_gripper_positions),
            int(self.gripper_binary_state)
        ))

    # ------------------------------------------------------------------ #
    #  Service Handlers (Unchanged)                                    #
    # ------------------------------------------------------------------ #

    def handle_set_recording(self, req):
        if req.data:
            self.collected_data = []
            self._discard_flag = False
            self.is_recording = True
            rospy.loginfo("=== Recording STARTED ===")
            return SetBoolResponse(True, "STARTED")
        else:
            self.is_recording = False
            path = self.save_data()
            if path:
                return SetBoolResponse(True, f"Saved to {path}")
            else:
                return SetBoolResponse(False, "Failed to save data (was any collected?)")

    def handle_discard(self, req):
        self._discard_flag = True
        return TriggerResponse(True, "Discarding...")

    def _do_discard(self):
        self.collected_data = []
        self.is_recording = False
        self._discard_flag = False
        rospy.logwarn("=== DISCARDED ===")

    # ------------------------------------------------------------------ #
    #  Save Data to .npz (Unchanged)                                   #
    # ------------------------------------------------------------------ #

    def save_data(self):
        if not self.collected_data: return None
        os.makedirs(self.save_dir, exist_ok=True)
        filename = os.path.join(self.save_dir, f"dual_cam_{time.strftime('%Y%m%d_%H%M%S')}.npz")
        
        # Save compressed arrays to disk
        np.savez_compressed(
            filename,
            images1=np.stack([d[0] for d in self.collected_data]),
            images2=np.stack([d[1] for d in self.collected_data]),
            joints=np.stack([d[2] for d in self.collected_data]),
            gripper_pos=np.stack([d[3] for d in self.collected_data]),
            gripper_state=np.array([d[4] for d in self.collected_data], dtype=np.int8)
        )
        rospy.loginfo(f"Saved recording with {len(self.collected_data)} samples to {filename}")
        self.collected_data = []  # Clear data after saving
        return filename

if __name__ == '__main__':
    try:
        collector = ControlledDataCollector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass