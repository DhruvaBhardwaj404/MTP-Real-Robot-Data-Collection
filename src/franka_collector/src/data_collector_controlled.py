#!/usr/bin/env python

import rospy
import numpy as np
import time
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge, CvBridgeError
from std_srvs.srv import SetBool, SetBoolResponse, Trigger, TriggerResponse
from franka_gripper.msg import GraspActionGoal, MoveActionGoal


class ControlledDataCollector:
    def __init__(self):
        rospy.init_node('franka_realsense_data_collector', anonymous=True)

        # --- State and Control ---
        self.is_recording = False
        self._discard_flag = False

        # --- ROS-to-Numpy Bridge ---
        self.bridge = CvBridge()

        # --- Latest sensor snapshots ---
        self.latest_image_data = None
        self.latest_joint_positions = None       # arm: float64 x7, radians
        self.latest_gripper_positions = None     # fingers: float64 x2, metres

        # --- Binary gripper state tracking ---
        # State: 0 = closing, 1 = opening, None = not yet determined
        self.gripper_binary_state = None
        self.previous_gripper_width = None       # total width = finger1 + finger2

        # Deadband: changes smaller than this (in metres) are treated as idle/noise.
        # 0.5 mm is a safe default for the Franka Hand.
        self.gripper_deadband_m = rospy.get_param('~gripper_deadband_m', 0.0005)

        # Directory where .npz files will be saved.
        # Defaults to the current working directory if not set.
        # Override at launch: rosrun your_pkg franka_data_collector.py _save_dir:=/data/replay
        self.save_dir = rospy.get_param('~save_dir', '.')

        # --- Data buffer ---
        # Each entry: (image_np, arm_joints_np, gripper_pos_np, gripper_binary_state)
        #   gripper_pos_np      : shape (2,)  — [finger_joint1, finger_joint2] in metres
        #   gripper_binary_state: int — 0 = closing, 1 = opening
        self.collected_data = []

        # --- Subscribers ---
        self.image_sub = rospy.Subscriber(
            "/camera/color/image_raw", Image,
            self.image_callback, queue_size=1
        )
        self.joint_sub = rospy.Subscriber(
            "/franka_state_controller/joint_states", JointState,
            self.joint_callback, queue_size=1
        )
        self.gripper_joint_sub = rospy.Subscriber(
            "/franka_gripper/joint_states", JointState,
            self.gripper_joint_callback, queue_size=1
        )
        # Snoop gripper action goals for optional context
        self.grasp_goal_sub = rospy.Subscriber(
            "/franka_gripper/grasp/goal", GraspActionGoal,
            self.grasp_goal_callback, queue_size=1
        )
        self.move_goal_sub = rospy.Subscriber(
            "/franka_gripper/move/goal", MoveActionGoal,
            self.move_goal_callback, queue_size=1
        )

        # --- Services ---
        self.control_service = rospy.Service(
            '/data_collector/set_recording', SetBool,
            self.handle_set_recording
        )
        self.discard_service = rospy.Service(
            '/data_collector/discard', Trigger,
            self.handle_discard
        )

        # --- Collection timer: 10 Hz ---
        self.collection_rate = 10.0
        rospy.Timer(rospy.Duration(1.0 / self.collection_rate), self.timer_callback)

        rospy.loginfo("ControlledDataCollector node started.")
        rospy.loginfo(f"  Gripper deadband: {self.gripper_deadband_m*1000:.1f} mm")
        rospy.loginfo(f"  Save directory:   {self.save_dir}")
        rospy.loginfo("  Start:   rosservice call /data_collector/set_recording \"data: true\"")
        rospy.loginfo("  Stop:    rosservice call /data_collector/set_recording \"data: false\"")
        rospy.loginfo("  Discard: rosservice call /data_collector/discard")

    # ------------------------------------------------------------------ #
    #  Subscriber Callbacks                                                #
    # ------------------------------------------------------------------ #

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_image_data = np.array(cv_image, dtype=np.uint8)
        except CvBridgeError as e:
            rospy.logerr(e)

    def joint_callback(self, msg):
        """Arm joints 1-7."""
        if len(msg.position) >= 7:
            self.latest_joint_positions = np.array(msg.position[:7], dtype=np.float64)

    def gripper_joint_callback(self, msg):
        """
        /franka_gripper/joint_states publishes panda_finger_joint1 and
        panda_finger_joint2 (prismatic, metres).

        This callback also runs the binary state update so it tracks state
        at the full gripper publish rate (~30 Hz), not just the 10 Hz collection
        rate — giving a more accurate velocity signal.
        """
        if len(msg.position) >= 2:
            new_positions = np.array(msg.position[:2], dtype=np.float64)
        elif len(msg.position) == 1:
            new_positions = np.array([msg.position[0], msg.position[0]], dtype=np.float64)
        else:
            return

        self.latest_gripper_positions = new_positions
        self._update_binary_gripper_state(new_positions)

    def grasp_goal_callback(self, msg):
        rospy.logdebug(
            f"Gripper GRASP goal: width={msg.goal.width:.4f} m  "
            f"speed={msg.goal.speed:.4f} m/s  force={msg.goal.force:.2f} N"
        )

    def move_goal_callback(self, msg):
        rospy.logdebug(
            f"Gripper MOVE goal: width={msg.goal.width:.4f} m  "
            f"speed={msg.goal.speed:.4f} m/s"
        )

    # ------------------------------------------------------------------ #
    #  Binary Gripper State Logic                                          #
    # ------------------------------------------------------------------ #

    def _update_binary_gripper_state(self, new_positions):
        """
        Called every time a new gripper joint message arrives.

        Logic:
          current_width = finger1 + finger2  (total gap in metres)
          delta         = current_width - previous_width

          delta < -deadband  →  fingers moving closer  →  CLOSING (0)
          delta > +deadband  →  fingers moving apart   →  OPENING (1)
          |delta| <= deadband →  idle/noise             →  keep last state

        On the very first call, we have no previous width yet, so we
        initialise previous_width and leave gripper_binary_state as None.
        It will be resolved on the next callback that produces a real delta.
        """
        current_width = float(new_positions[0] + new_positions[1])

        if self.previous_gripper_width is None:
            # First ever reading — just seed the previous width
            self.previous_gripper_width = current_width
            return

        delta = current_width - self.previous_gripper_width

        if delta < -self.gripper_deadband_m:
            self.gripper_binary_state = 0   # closing
        elif delta > self.gripper_deadband_m:
            self.gripper_binary_state = 1   # opening
        # else: idle — state stays as whatever it last was

        self.previous_gripper_width = current_width

    # ------------------------------------------------------------------ #
    #  Timer — core collection loop                                        #
    # ------------------------------------------------------------------ #

    def timer_callback(self, event):
        if not self.is_recording:
            return

        if self._discard_flag:
            self._do_discard()
            return

        if self.latest_image_data is None or self.latest_joint_positions is None:
            rospy.logwarn_throttle(5, "Recording active — waiting for arm/image data...")
            return

        if self.latest_gripper_positions is None:
            rospy.logwarn_throttle(5, "Recording active — waiting for gripper data "
                                      "on /franka_gripper/joint_states ...")
            return

        if self.gripper_binary_state is None:
            # We have gripper positions but not enough history yet for a delta.
            # This resolves after the second gripper message, so it's very transient.
            rospy.logwarn_throttle(2, "Waiting for gripper motion history to determine state...")
            return

        self.collected_data.append((
            np.copy(self.latest_image_data),
            np.copy(self.latest_joint_positions),
            np.copy(self.latest_gripper_positions),
            int(self.gripper_binary_state)
        ))

        state_str = "OPENING" if self.gripper_binary_state == 1 else "CLOSING"
        rospy.loginfo_throttle(
            1,
            f"Recording... {len(self.collected_data)} samples  |  "
            f"gripper: {self.latest_gripper_positions.sum()*1000:.1f} mm total  |  "
            f"state: {state_str}"
        )

    # ------------------------------------------------------------------ #
    #  Service Handlers                                                    #
    # ------------------------------------------------------------------ #

    def handle_set_recording(self, req):
        response = SetBoolResponse()
        if req.data:
            if self.is_recording:
                response.success = False
                response.message = "Already recording. Call /data_collector/discard to cancel first."
            else:
                self.collected_data = []
                self._discard_flag = False
                self.is_recording = True
                response.success = True
                response.message = "Recording STARTED."
                rospy.loginfo("=== Recording STARTED ===")
        else:
            if not self.is_recording:
                response.success = False
                response.message = "Not currently recording."
            else:
                self.is_recording = False
                saved_path = self.save_data()
                if saved_path:
                    response.success = True
                    response.message = f"Recording stopped. Data saved to: {saved_path}"
                else:
                    response.success = False
                    response.message = "Recording stopped, but no data was collected — nothing saved."
        return response

    def handle_discard(self, req):
        response = TriggerResponse()
        if not self.is_recording:
            response.success = False
            response.message = "Not currently recording. Nothing to discard."
        else:
            self._discard_flag = True
            n = len(self.collected_data)
            response.success = True
            response.message = f"Discard requested — {n} buffered samples will be dropped."
            rospy.logwarn(f"=== DISCARD requested — {n} samples will be thrown away ===")
        return response

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _do_discard(self):
        n = len(self.collected_data)
        self.collected_data = []
        self.is_recording = False
        self._discard_flag = False
        rospy.logwarn(f"=== Recording DISCARDED — {n} samples dropped. Ready for a new recording. ===")

    def save_data(self):
        """
        Saves all collected data to a compressed .npz file.

        Arrays saved:
          images        : (N, H, W, 3)  uint8   — BGR frames
          joints        : (N, 7)        float64 — arm joint positions (rad)
          gripper_pos   : (N, 2)        float64 — [finger1, finger2] in metres
          gripper_state : (N,)          int8    — binary: 0=closing, 1=opening
        """
        if not self.collected_data:
            rospy.logwarn("No data collected — nothing to save.")
            return None

        n = len(self.collected_data)
        rospy.loginfo(f"Saving {n} data points...")

        images_array        = np.stack([d[0] for d in self.collected_data])
        joints_array        = np.stack([d[1] for d in self.collected_data])
        gripper_pos_array   = np.stack([d[2] for d in self.collected_data])
        gripper_state_array = np.array([d[3] for d in self.collected_data], dtype=np.int8)

        import os
        os.makedirs(self.save_dir, exist_ok=True)
        filename = os.path.join(
            self.save_dir,
            f"realsense_franka_data_{time.strftime('%Y%m%d_%H%M%S')}.npz"
        )
        np.savez_compressed(
            filename,
            images=images_array,
            joints=joints_array,
            gripper_pos=gripper_pos_array,
            gripper_state=gripper_state_array
        )

        closing_count = int((gripper_state_array == 0).sum())
        opening_count = int((gripper_state_array == 1).sum())
        rospy.loginfo(
            f"Saved → {filename}\n"
            f"  images        : {images_array.shape}  dtype={images_array.dtype}\n"
            f"  joints        : {joints_array.shape}  dtype={joints_array.dtype}\n"
            f"  gripper_pos   : {gripper_pos_array.shape}  dtype={gripper_pos_array.dtype}\n"
            f"  gripper_state : {gripper_state_array.shape}  dtype={gripper_state_array.dtype} "
            f"  (closing={closing_count}, opening={opening_count})"
        )
        self.collected_data = []
        return filename


# ------------------------------------------------------------------ #
#  Entry point                                                         #
# ------------------------------------------------------------------ #

if __name__ == '__main__':
    try:
        collector = ControlledDataCollector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass