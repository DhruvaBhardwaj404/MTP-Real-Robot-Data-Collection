#!/usr/bin/env python3

import rospy
import numpy as np
import time
import cv2
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge, CvBridgeError
from std_srvs.srv import SetBool, SetBoolResponse

# Define a small velocity threshold to detect movement vs. stationary state
# Velocities below this magnitude are considered stationary/zero.
VELOCITY_THRESHOLD = 1e-3


class ControlledDataCollector:
    def __init__(self):
        rospy.init_node('franka_realsense_data_collector', anonymous=True)

        # --- State and Control ---
        self.is_recording = False  # Main state flag

        # --- ROS-to-Numpy Bridge ---
        self.bridge = CvBridge()

        # --- Data Storage ---
        self.latest_image_data = None  # Camera 1 Image (np.uint8)
        self.latest_image_data_2 = None  # Camera 2 Image (np.uint8)
        self.latest_joint_positions = None  # Arm Joints (np.float64)
        self.latest_gripper_positions = None  # Gripper Joints (np.float64)
        # NEW: Gripper movement state (string)
        self.latest_gripper_state_str = "unknown"

        # List to store collected (img1, img2, arm_joints, gripper_joints, gripper_state_str) tuples
        self.collected_data = []

        # --- ROS Subscribers (Topics remain active regardless of recording state) ---
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback, queue_size=1)
        self.image_sub_2 = rospy.Subscriber("/camera_2/color/image_raw", Image, self.image_callback_2, queue_size=1)

        self.joint_sub = rospy.Subscriber("/franka_state_controller/joint_states", JointState, self.joint_callback,
                                          queue_size=1)
        # UPDATED: We need the full JointState message in the gripper callback to access velocity
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
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_image_data = np.array(cv_image, dtype=np.uint8)
        except CvBridgeError as e:
            rospy.logerr(e)

    def image_callback_2(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_image_data_2 = np.array(cv_image, dtype=np.uint8)
        except CvBridgeError as e:
            rospy.logerr(e)

    def joint_callback(self, msg):
        """Stores the latest 7 Franka arm joint positions."""
        if len(msg.position) >= 7:
            self.latest_joint_positions = np.array(msg.position[:7], dtype=np.float64)

    def gripper_callback(self, msg):
        """
        Stores the latest 2 Franka gripper joint positions AND infers the state
        (opening, closing, or stationary) from the velocity.
        """
        # 1. Store Positions
        if len(msg.position) >= 2:
            self.latest_gripper_positions = np.array(msg.position[:2], dtype=np.float64)

        # 2. Infer State from Velocity
        if len(msg.velocity) >= 2:
            gripper_vel = msg.velocity[0]

            if gripper_vel > VELOCITY_THRESHOLD:
                self.latest_gripper_state_str = "opening"
            elif gripper_vel < -VELOCITY_THRESHOLD:
                self.latest_gripper_state_str = "closing"
            else:
                self.latest_gripper_state_str = "stationary"
        else:
            self.latest_gripper_state_str = "unknown"

    # ----------------------------------------------------------------------
    # ## ⏱️ Core Logic (10 Hz Timer)
    # ----------------------------------------------------------------------

    def timer_callback(self, event):
        """Runs at 10 Hz. Collects data if recording is True and all topics have fresh data."""
        if not self.is_recording:
            return

        # Check for synchronization (now includes the gripper state string)
        if (self.latest_image_data is not None and
                self.latest_image_data_2 is not None and
                self.latest_joint_positions is not None and
                self.latest_gripper_positions is not None and
                self.latest_gripper_state_str != "unknown"):  # Ensure state has been set

            # Copy data for storage
            image_copy_1 = np.copy(self.latest_image_data)
            image_copy_2 = np.copy(self.latest_image_data_2)
            joints_copy = np.copy(self.latest_joint_positions)
            gripper_copy = np.copy(self.latest_gripper_positions)
            gripper_state_copy = self.latest_gripper_state_str  # String copy is fine

            # Store the synchronized data point (5 elements now)
            self.collected_data.append((image_copy_1, image_copy_2, joints_copy, gripper_copy, gripper_state_copy))
            rospy.loginfo(
                f"Data point collected: {len(self.collected_data)} total. Gripper State: {gripper_state_copy}")
        else:
            rospy.logwarn_throttle(5,
                                   "Recording, but waiting for initial data from all topics (including Gripper State)...")

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

        # ... (rest of the service handler logic remains the same)
        elif req.data and self.is_recording:
            response.success = True
            response.message = "Already recording."

        elif not req.data and not self.is_recording:
            response.success = True
            response.message = "Not currently recording. No action taken."

        return response


    def save_data(self):
        """Combines, resizes images, and saves all collected data to a NumPy .npz file."""
        if not self.collected_data:
            rospy.logwarn("Recording stopped, but no data was collected. No file saved.")
            return

        rospy.loginfo(f"Saving {len(self.collected_data)} data points...")

        # Separate the raw data (UPDATED to pull 5 elements)
        raw_images_1 = [data[0] for data in self.collected_data]
        raw_images_2 = [data[1] for data in self.collected_data]
        joints = [data[2] for data in self.collected_data]
        gripper_pos = [data[3] for data in self.collected_data]
        gripper_state_list = [data[4] for data in self.collected_data]


        resized_images_1 = []
        resized_images_2 = []
        target_size = (84, 84)

        rospy.loginfo(f"Resizing {len(raw_images_1)} images to {target_size}...")


        for img in raw_images_1:
            resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            resized_images_1.append(resized_img)


        for img in raw_images_2:
            resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            resized_images_2.append(resized_img)

        # Convert the lists to final NumPy arrays
        images_array_1 = np.stack(resized_images_1)
        images_array_2 = np.stack(resized_images_2)
        joints_array = np.stack(joints)
        gripper_pos_array = np.stack(gripper_pos)

        # Convert list of strings to a NumPy object array
        gripper_state_array = np.array(gripper_state_list, dtype=object)  # NEW ARRAY

        # Save to file using the requested descriptive keys
        filename = f"realsense_franka_data_{time.strftime('%Y%m%d_%H%M%S')}.npz"
        np.savez_compressed(
            filename,
            camera_image=images_array_1,
            camera_wrist_image=images_array_2,
            arm_joints=joints_array,
            gripper_positions=gripper_pos_array,
            gripper_state = gripper_state_array
        )

        # Log the final shape for verification
        rospy.loginfo(f"Successfully saved data to {filename}")
        rospy.loginfo(f"Final camera_image Array Shape: {images_array_1.shape}")
        rospy.loginfo(f"Final gripper_state Array Shape: {gripper_state_array.shape}")
        rospy.loginfo(f"Final Arm Joints Array Shape: {joints_array.shape}")

        self.collected_data = []  # Clear the buffer after saving


if __name__ == '__main__':
    try:
        collector = ControlledDataCollector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass