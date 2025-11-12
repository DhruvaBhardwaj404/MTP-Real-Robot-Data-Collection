#!/usr/bin/env python3

import rospy
import numpy as np
import time
import cv2
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge, CvBridgeError
from std_srvs.srv import SetBool, SetBoolResponse


class ControlledDataCollector:
    def __init__(self):
        rospy.init_node('franka_realsense_data_collector', anonymous=True)

        # --- State and Control ---
        self.is_recording = False  # Main state flag

        # --- ROS-to-Numpy Bridge ---
        self.bridge = CvBridge()

        # --- Data Storage ---
        self.latest_image_data = None
        self.latest_joint_positions = None
        self.latest_gripper_positions = None
        # List to store collected (image_np, arm_joints_np, gripper_joints_np) tuples
        self.collected_data = []

        # --- ROS Subscribers (Topics remain active regardless of recording state) ---
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback, queue_size=1)
        self.joint_sub = rospy.Subscriber("/franka_state_controller/joint_states", JointState, self.joint_callback,
                                          queue_size=1)
        self.gripper_sub = rospy.Subscriber("/franka_gripper/joint_states", JointState, self.gripper_callback,
                                            queue_size=1)

        # --- ROS Service (Control Mechanism) ---
        self.control_service = rospy.Service(
            '/data_collector/set_recording',
            SetBool,
            self.handle_set_recording
        )

        # --- Collection Rate Timer (10 Hz) ---
        self.collection_rate = 10.0  # Hz
        rospy.Timer(rospy.Duration(1.0 / self.collection_rate), self.timer_callback)

        rospy.loginfo("ControlledDataCollector node started. Waiting for service calls...")

    # ----------------------------------------------------------------------
    # ## 📥 Subscriber Callbacks
    # ----------------------------------------------------------------------

    def image_callback(self, msg):
        """Stores the latest image data as a BGR NumPy array."""
        try:
            # Use bgr8 encoding for standard color image processing
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_image_data = np.array(cv_image, dtype=np.uint8)
        except CvBridgeError as e:
            rospy.logerr(e)

    def joint_callback(self, msg):
        """Stores the latest 7 Franka arm joint positions."""
        # This assumes the first 7 positions are the arm joints (q1-q7)
        if len(msg.position) >= 7:
            self.latest_joint_positions = np.array(msg.position[:7], dtype=np.float64)

    def gripper_callback(self, msg):
        """Stores the latest 2 Franka gripper joint positions."""
        # The Franka gripper usually has 2 joints (panda_finger_joint1, panda_finger_joint2)
        if len(msg.position) >= 2:
            self.latest_gripper_positions = np.array(msg.position[:2], dtype=np.float64)

    # ----------------------------------------------------------------------
    # ## ⏱️ Core Logic (10 Hz Timer)
    # ----------------------------------------------------------------------

    def timer_callback(self, event):
        """Runs at 10 Hz. Only collects data if self.is_recording is True and all topics have fresh data."""
        if not self.is_recording:
            return

        # Check for synchronized data from all three sources
        if (self.latest_image_data is not None and
                self.latest_joint_positions is not None and
                self.latest_gripper_positions is not None):

            # Copy data for storage to prevent modification by callbacks
            image_copy = np.copy(self.latest_image_data)
            joints_copy = np.copy(self.latest_joint_positions)
            gripper_copy = np.copy(self.latest_gripper_positions)

            # Store the synchronized data point
            self.collected_data.append((image_copy, joints_copy, gripper_copy))
            rospy.loginfo(f"Data point collected: {len(self.collected_data)} total.")
        else:
            rospy.logwarn_throttle(5,
                                   "Recording, but waiting for initial data from topics (Image, Arm Joints, or Gripper Joints)...")

    # ----------------------------------------------------------------------
    # ## ⚙️ Service Handler
    # ----------------------------------------------------------------------

    def handle_set_recording(self, req):
        """Handles service calls to start or stop recording."""
        response = SetBoolResponse()

        if req.data and not self.is_recording:
            # Start Recording
            self.collected_data = []  # Clear previous data buffer
            self.is_recording = True
            response.success = True
            response.message = "Recording started."
            rospy.loginfo("Recording has STARTED.")

        elif not req.data and self.is_recording:
            # Stop Recording and Save
            self.is_recording = False
            self.save_data()  # Perform the saving operation
            response.success = True
            response.message = "Recording stopped and data saved."

        elif req.data and self.is_recording:
            response.success = True
            response.message = "Already recording."

        elif not req.data and not self.is_recording:
            response.success = True
            response.message = "Not currently recording. No action taken."

        return response

    # ----------------------------------------------------------------------
    # ## 💾 Saving Function (with Resize)
    # ----------------------------------------------------------------------

    def save_data(self):
        """Combines, resizes images, and saves all collected data to a NumPy .npz file."""
        if not self.collected_data:
            rospy.logwarn("Recording stopped, but no data was collected. No file saved.")
            return

        rospy.loginfo(f"Saving {len(self.collected_data)} data points...")

        # Separate the raw data
        raw_images = [data[0] for data in self.collected_data]
        joints = [data[1] for data in self.collected_data]
        gripper = [data[2] for data in self.collected_data]

        # --- Image Preprocessing: Resize to 84x84x3 ---
        resized_images = []
        target_size = (84, 84)

        rospy.loginfo(f"Resizing {len(raw_images)} images to {target_size}...")
        for img in raw_images:
            # Resize image using cv2.INTER_AREA (best for downsampling)
            resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            resized_images.append(resized_img)

        # Convert the lists to final NumPy arrays
        images_array = np.stack(resized_images)
        joints_array = np.stack(joints)
        gripper_array = np.stack(gripper)

        # Save to file
        filename = f"realsense_franka_data_{time.strftime('%Y%m%d_%H%M%S')}.npz"
        np.savez_compressed(
            filename,
            images=images_array,
            arm_joints=joints_array,
            gripper_joints=gripper_array
        )

        # Log the final shape for verification
        rospy.loginfo(f"Successfully saved data to {filename}")
        rospy.loginfo(f"Final Image Array Shape: {images_array.shape} (N, 84, 84, 3)")
        rospy.loginfo(f"Final Arm Joints Array Shape: {joints_array.shape}")

        self.collected_data = []  # Clear the buffer after saving


if __name__ == '__main__':
    try:
        collector = ControlledDataCollector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass