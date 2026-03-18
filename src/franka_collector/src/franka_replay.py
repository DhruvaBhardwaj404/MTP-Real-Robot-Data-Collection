#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import actionlib
import sys
from scipy.signal import savgol_filter

from sensor_msgs.msg import JointState
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from franka_gripper.msg import MoveAction, MoveGoal, GraspAction, GraspGoal
from std_srvs.srv import SetBool, Trigger

# --- Constants ---
ARM_JOINT_NAMES = [
    'panda_joint1', 'panda_joint2', 'panda_joint3',
    'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7'
]

ARM_ACTION_SERVER = '/position_joint_trajectory_controller/follow_joint_trajectory'
GRIPPER_MOVE_SERVER  = '/franka_gripper/move'
GRIPPER_GRASP_SERVER = '/franka_gripper/grasp'

GRIPPER_SPEED     = 0.1
GRIPPER_FORCE     = 20.0  
GRIPPER_EPSILON   = 0.01

class FrankaReplayer:
    def __init__(self, npz_path):
        rospy.init_node('franka_replayer', anonymous=True)

        # --- Load data ---
        try:
            data = np.load(npz_path)
            self.joints_array        = data['joints']
            self.gripper_pos_array   = data['gripper_pos']
            self.gripper_state_array = data['gripper_state']
            self.n_steps             = self.joints_array.shape[0]

            window_len = 7 
            poly_order = 2

            # Smooth the joint data to reduce jerk
            for i in range(7):
                self.joints_array[:, i] = savgol_filter(self.joints_array[:, i], window_len, poly_order)

        except Exception as e:
            rospy.logerr(f"Failed to load NPZ file: {e}")
            sys.exit(1)

        # --- State Tracking ---
        self.current_joint_positions = None
        self.joint_sub = rospy.Subscriber(
            '/franka_state_controller/joint_states', JointState,
            self._joint_callback, queue_size=1
        )

        # --- Action Clients ---
        self.arm_client = actionlib.SimpleActionClient(ARM_ACTION_SERVER, FollowJointTrajectoryAction)
        self.move_client  = actionlib.SimpleActionClient(GRIPPER_MOVE_SERVER,  MoveAction)
        self.grasp_client = actionlib.SimpleActionClient(GRIPPER_GRASP_SERVER, GraspAction)

        rospy.loginfo("Connecting to Action Servers...")
        if not self.arm_client.wait_for_server(rospy.Duration(5.0)):
            rospy.logerr("Arm server missing!")
            sys.exit(1)
        
        # --- Service Proxies ---
        rospy.wait_for_service('/data_collector/set_recording', timeout=5.0)
        self._svc_set_recording = rospy.ServiceProxy('/data_collector/set_recording', SetBool)

        # --- Timing ---
        self.collection_hz = 10.0
        self.replay_rate   = rospy.Rate(self.collection_hz) 

    def _joint_callback(self, msg):
        if len(msg.position) >= 7:
            self.current_joint_positions = np.array(msg.position[:7], dtype=np.float64)

    def _move_to_start(self):
        """Slowly moves the robot to the first frame of the recording."""
        rospy.loginfo("Moving to initial recorded position...")
        first_pose = self.joints_array[0]
        
        point = JointTrajectoryPoint()
        point.positions = first_pose.tolist()
        point.time_from_start = rospy.Duration(4.0) 

        traj = JointTrajectory()
        traj.joint_names = ARM_JOINT_NAMES
        traj.points = [point]

        goal = FollowJointTrajectoryGoal()
        goal.trajectory = traj
        
        self.arm_client.send_goal_and_wait(goal)
        rospy.loginfo("Robot in position. Ready for replay.")

    def _send_full_trajectory(self):
        """Packages the entire recording into a single, smooth trajectory."""
        rospy.loginfo("Packaging full trajectory for smooth execution...")
        goal = FollowJointTrajectoryGoal()
        traj = JointTrajectory()
        traj.joint_names = ARM_JOINT_NAMES

        skip_interval = 5  # Jump every 5 frames (0.5 seconds)
    
        for i in range(0, self.n_steps, skip_interval):
            point = JointTrajectoryPoint()
            point.positions = self.joints_array[i].tolist()
            # Keep original time: Step 0 is 0.0s, Step 5 is 0.5s, Step 10 is 1.0s
            point.time_from_start = rospy.Duration.from_sec(i * 0.1)
            traj.points.append(point)

        if (self.n_steps - 1) % skip_interval != 0:
            last_idx = self.n_steps - 1
            last_point = JointTrajectoryPoint()
            last_point.positions = self.joints_array[last_idx].tolist()
            last_point.time_from_start = rospy.Duration.from_sec(last_idx * 0.1)
            traj.points.append(last_point)
            rospy.loginfo("Added final goal point to trajectory.")

        goal.trajectory = traj
        # Send goal asynchronously so we can handle the gripper in our own loop
        self.arm_client.send_goal(goal)

    def _send_gripper_step(self, step_idx, current_state):
        """Dispatches a gripper command only when the state toggles."""
        target_width = float(self.gripper_pos_array[step_idx].sum())

        if current_state == 0: # Closing/Grasping
            goal = GraspGoal(width=target_width, speed=GRIPPER_SPEED, force=GRIPPER_FORCE)
            goal.epsilon.inner = goal.epsilon.outer = GRIPPER_EPSILON
            self.grasp_client.send_goal(goal)
            rospy.loginfo(f"[{step_idx}/{self.n_steps}] Gripper: GRASPING (width={target_width:.4f})")
        else: # Opening/Moving
            self.move_client.send_goal(MoveGoal(width=target_width, speed=GRIPPER_SPEED))
            rospy.loginfo(f"[{step_idx}/{self.n_steps}] Gripper: MOVING/OPENING (width={target_width:.4f})")

    def run(self):
        # 1. Wait for robot feedback
        while self.current_joint_positions is None and not rospy.is_shutdown():
            rospy.sleep(0.1)

        # 2. Safe Transition to start pose
        self._move_to_start()

        # Prompt user before beginning the automated motion
        input("Press Enter to begin continuous smooth replay...")

        # 3. Start Data Collector
        try:
            self._svc_set_recording(True)
        except Exception as e:
            rospy.logwarn(f"Could not trigger Data Collector: {e}. Replaying anyway...")

        # 4. Send the full arm motion
        self._send_full_trajectory()

        # 5. Replay Loop (Gripper tracking)
        rospy.loginfo(f"Replaying {self.n_steps} steps smoothly. Press Ctrl+C to stop.")
        
        last_state = -1 # Initialize to -1 to force the first gripper command to trigger

        for step_idx in range(self.n_steps):
            if rospy.is_shutdown(): 
                self.arm_client.cancel_all_goals()
                break

            current_state = int(self.gripper_state_array[step_idx])
            
            # Only trigger an action client if the binary state toggles
            if current_state != last_state:
                self._send_gripper_step(step_idx, current_state)
                last_state = current_state

            if step_idx % 50 == 0 and step_idx > 0:
                rospy.loginfo(f"Progress: {100.0 * step_idx / self.n_steps:.1f}%")

            # Sleep to keep the gripper commands synced with the arm trajectory execution
            self.replay_rate.sleep()

        # 6. Cleanup
        # Wait a moment to ensure the trajectory fully completes
        self.arm_client.wait_for_result(rospy.Duration(2.0))
        rospy.loginfo("Replay finished. Saving data...")
        
        try:
            self._svc_set_recording(False)
        except:
            pass

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 franka_replay.py <data.npz>")
        sys.exit(1)

    replayer = FrankaReplayer(sys.argv[1])
    replayer.run()