#!/usr/bin/env python

import rospy
import numpy as np
import actionlib
import sys

from sensor_msgs.msg import JointState
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from franka_gripper.msg import MoveAction, MoveGoal, GraspAction, GraspGoal
from std_srvs.srv import SetBool, Trigger


# ------------------------------------------------------------------ #
#  Constants                                                           #
# ------------------------------------------------------------------ #

ARM_JOINT_NAMES = [
    'panda_joint1', 'panda_joint2', 'panda_joint3',
    'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7'
]

ARM_ACTION_SERVER    = '/position_joint_trajectory_controller/follow_joint_trajectory'
GRIPPER_MOVE_SERVER  = '/franka_gripper/move'
GRIPPER_GRASP_SERVER = '/franka_gripper/grasp'

GRIPPER_SPEED     = 0.1
GRIPPER_FORCE     = 10.0
GRIPPER_EPSILON   = 0.01

DEVIATION_WARN_THRESHOLD = 0.05   # radians (~3 degrees)


class FrankaReplayer:
    def __init__(self, npz_path):
        rospy.init_node('franka_replayer', anonymous=True)

        # --- Load data ---
        rospy.loginfo(f"Loading data from: {npz_path}")
        data = np.load(npz_path)

        self.joints_array        = data['joints']
        self.gripper_pos_array   = data['gripper_pos']
        self.gripper_state_array = data['gripper_state']
        self.n_steps             = self.joints_array.shape[0]

        rospy.loginfo(
            f"Loaded {self.n_steps} steps  |  "
            f"arm: {self.joints_array.shape}  |  "
            f"gripper_pos: {self.gripper_pos_array.shape}  |  "
            f"gripper_state: {self.gripper_state_array.shape}"
        )

        # --- Current joint state for deviation check ---
        self.current_joint_positions = None
        self.joint_sub = rospy.Subscriber(
            '/franka_state_controller/joint_states', JointState,
            self._joint_callback, queue_size=1
        )

        # --- Arm action client ---
        rospy.loginfo(f"Connecting to arm action server: {ARM_ACTION_SERVER}")
        self.arm_client = actionlib.SimpleActionClient(
            ARM_ACTION_SERVER, FollowJointTrajectoryAction
        )
        if not self.arm_client.wait_for_server(rospy.Duration(10.0)):
            rospy.logerr("Arm action server not available.")
            sys.exit(1)
        rospy.loginfo("Arm action server connected.")

        # --- Gripper action clients ---
        self.move_client  = actionlib.SimpleActionClient(GRIPPER_MOVE_SERVER,  MoveAction)
        self.grasp_client = actionlib.SimpleActionClient(GRIPPER_GRASP_SERVER, GraspAction)
        if not self.move_client.wait_for_server(rospy.Duration(5.0)):
            rospy.logerr("Gripper move server not available.")
            sys.exit(1)
        if not self.grasp_client.wait_for_server(rospy.Duration(5.0)):
            rospy.logerr("Gripper grasp server not available.")
            sys.exit(1)
        rospy.loginfo("Gripper action servers connected.")

        # --- Data collector service proxies ---
        # These will be called to start/stop recording during replay.
        # The data collector node must already be running.
        rospy.loginfo("Waiting for data collector services...")
        rospy.wait_for_service('/data_collector/set_recording', timeout=10.0)
        rospy.wait_for_service('/data_collector/discard',       timeout=10.0)
        self._svc_set_recording = rospy.ServiceProxy('/data_collector/set_recording', SetBool)
        self._svc_discard       = rospy.ServiceProxy('/data_collector/discard',       Trigger)
        rospy.loginfo("Data collector services found.")

        # --- Replay timing ---
        self.replay_rate   = rospy.Rate(10.0)
        self.step_duration = rospy.Duration(0.15)

    # ------------------------------------------------------------------ #
    #  Subscriber                                                          #
    # ------------------------------------------------------------------ #

    def _joint_callback(self, msg):
        if len(msg.position) >= 7:
            self.current_joint_positions = np.array(msg.position[:7], dtype=np.float64)

    # ------------------------------------------------------------------ #
    #  Data collector control                                              #
    # ------------------------------------------------------------------ #

    def _start_recording(self):
        """Calls the collector service to begin a new recording."""
        try:
            resp = self._svc_set_recording(True)
            if resp.success:
                rospy.loginfo("Re-recording STARTED by replayer.")
            else:
                rospy.logwarn(f"Collector refused start: {resp.message}")
        except rospy.ServiceException as e:
            rospy.logerr(f"Failed to call set_recording(True): {e}")

    def _stop_recording(self):
        """Calls the collector service to stop and save the re-recording."""
        try:
            resp = self._svc_set_recording(False)
            if resp.success:
                rospy.loginfo(f"Re-recording STOPPED and saved: {resp.message}")
            else:
                rospy.logwarn(f"Collector refused stop: {resp.message}")
        except rospy.ServiceException as e:
            rospy.logerr(f"Failed to call set_recording(False): {e}")

    def _discard_recording(self):
        """Discards the in-progress re-recording (called on abort)."""
        try:
            resp = self._svc_discard()
            rospy.logwarn(f"Re-recording discarded: {resp.message}")
        except rospy.ServiceException as e:
            rospy.logerr(f"Failed to call discard: {e}")

    # ------------------------------------------------------------------ #
    #  Arm helpers                                                         #
    # ------------------------------------------------------------------ #

    def _send_arm_step(self, joint_positions):
        point = JointTrajectoryPoint()
        point.positions      = joint_positions.tolist()
        point.velocities     = [0.0] * 7
        point.time_from_start = self.step_duration

        traj = JointTrajectory()
        traj.joint_names = ARM_JOINT_NAMES
        traj.points      = [point]

        goal = FollowJointTrajectoryGoal()
        goal.trajectory = traj
        self.arm_client.send_goal(goal)

    def _check_deviation(self, step_idx, recorded_joints):
        if self.current_joint_positions is None:
            return
        errors      = np.abs(self.current_joint_positions - recorded_joints)
        worst_joint = int(np.argmax(errors))
        worst_error = float(errors[worst_joint])
        if worst_error > DEVIATION_WARN_THRESHOLD:
            rospy.logwarn(
                f"Step {step_idx}: deviation on panda_joint{worst_joint+1} = "
                f"{np.degrees(worst_error):.2f} deg"
            )

    # ------------------------------------------------------------------ #
    #  Gripper helpers                                                     #
    # ------------------------------------------------------------------ #

    def _send_gripper_step(self, step_idx):
        current_state = int(self.gripper_state_array[step_idx])
        target_width  = float(self.gripper_pos_array[step_idx].sum())

        send_new = False
        if not hasattr(self, '_last_gripper_state'):
            send_new = True
        elif self._last_gripper_state != current_state:
            send_new = True
        elif abs(target_width - self._last_gripper_width) > 0.001:
            send_new = True

        if not send_new:
            return

        self._last_gripper_state = current_state
        self._last_gripper_width = target_width

        if current_state == 0:
            goal               = GraspGoal()
            goal.width         = target_width
            goal.speed         = GRIPPER_SPEED
            goal.force         = GRIPPER_FORCE
            goal.epsilon.inner = GRIPPER_EPSILON
            goal.epsilon.outer = GRIPPER_EPSILON
            self.grasp_client.send_goal(goal)
        else:
            goal       = MoveGoal()
            goal.width = target_width
            goal.speed = GRIPPER_SPEED
            self.move_client.send_goal(goal)

    # ------------------------------------------------------------------ #
    #  Main replay loop                                                    #
    # ------------------------------------------------------------------ #

    def run(self):
        rospy.loginfo("=== Waiting for initial joint state feedback ===")
        while self.current_joint_positions is None and not rospy.is_shutdown():
            rospy.sleep(0.1)

        rospy.loginfo(f"=== Starting replay of {self.n_steps} steps at 10 Hz ===")
        rospy.loginfo("Press Ctrl+C to abort.")

        aborted = False

        for step_idx in range(self.n_steps):
            if rospy.is_shutdown():
                rospy.logwarn("ROS shutdown — aborting replay.")
                aborted = True
                break

            recorded_joints = self.joints_array[step_idx]

            # 1. Send arm command
            self._send_arm_step(recorded_joints)

            # 2. Send gripper command
            self._send_gripper_step(step_idx)

            # 3. After the very first step is dispatched, start re-recording.
            #    This ensures the robot is already moving when recording begins,
            #    matching the behaviour of the original recording session.
            if step_idx == 0:
                self._start_recording()

            # 4. Deviation check
            self._check_deviation(step_idx, recorded_joints)

            # 5. Progress log every 50 steps
            if step_idx % 50 == 0:
                pct = 100.0 * step_idx / self.n_steps
                rospy.loginfo(f"Replay progress: step {step_idx}/{self.n_steps}  ({pct:.1f}%)")

            # 6. Pace at 10 Hz
            self.replay_rate.sleep()

        if aborted:
            # Replay was cut short — discard the incomplete re-recording
            rospy.logwarn("Replay aborted — discarding incomplete re-recording.")
            self._discard_recording()
        else:
            # Replay finished cleanly — stop and save the re-recording
            rospy.loginfo("=== Replay complete — stopping re-recording ===")
            self._stop_recording()

        self.arm_client.cancel_all_goals()


# ------------------------------------------------------------------ #
#  Entry point                                                         #
# ------------------------------------------------------------------ #

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: rosrun <pkg> franka_replay.py <path_to_npz_file>")
        sys.exit(1)

    try:
        replayer = FrankaReplayer(sys.argv[1])
        replayer.run()
    except rospy.ROSInterruptException:
        pass