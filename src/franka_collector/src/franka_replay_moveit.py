#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import sys
import actionlib

import moveit_commander
from moveit_msgs.msg import RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

from sensor_msgs.msg import JointState
from franka_gripper.msg import MoveAction, MoveGoal, GraspAction, GraspGoal
from std_srvs.srv import SetBool

GRIPPER_MOVE_SERVER  = '/franka_gripper/move'
GRIPPER_GRASP_SERVER = '/franka_gripper/grasp'

GRIPPER_SPEED   = 0.05
GRIPPER_FORCE   = 5.0
GRIPPER_EPSILON = 0.02

DT = 0.1  # seconds per recorded step (10 Hz recording)


class FrankaReplayerMoveIt:
    def __init__(self, npz_path):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('franka_replayer_moveit', anonymous=True)

        # ------------------------------------------------------------------ #
        #  Load data — no smoothing needed, velocities are from hardware     #
        # ------------------------------------------------------------------ #
        try:
            data = np.load(npz_path)
            self.joints_array        = data['joints'].astype(np.float64)
            self.velocities_array    = data['joint_velocities'].astype(np.float64)  # real measured
            self.gripper_pos_array   = data['gripper_pos']
            self.gripper_state_array = data['gripper_state']
            self.n_steps             = self.joints_array.shape[0]
        except KeyError:
            rospy.logerr(
                "Key 'joint_velocities' not found in NPZ. "
                "Please re-record with the updated data collector script."
            )
            sys.exit(1)
        except Exception as e:
            rospy.logerr(f"Failed to load NPZ file: {e}")
            sys.exit(1)

        # Accelerations: np.gradient on hardware-measured velocities is fine
        # because the hardware velocity signal is already filtered/smooth
        self.accels_array = np.gradient(self.velocities_array, DT, axis=0)
        
        rospy.loginfo(
            f"Loaded {self.n_steps} steps. "
            f"Max joint vel: {np.abs(self.velocities_array).max():.3f} rad/s"
        )

        # ------------------------------------------------------------------ #
        #  MoveIt                                                             #
        # ------------------------------------------------------------------ #
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.arm   = moveit_commander.MoveGroupCommander("panda_arm")

        self.arm.set_max_velocity_scaling_factor(0.1)
        self.arm.set_max_acceleration_scaling_factor(0.1)
        self.arm.set_goal_joint_tolerance(0.01)

        self.gripper_state_closed = False

        rospy.loginfo(f"Planning frame : {self.arm.get_planning_frame()}")
        rospy.loginfo(f"End-effector   : {self.arm.get_end_effector_link()}")

        # ------------------------------------------------------------------ #
        #  Gripper clients                                                    #
        # ------------------------------------------------------------------ #
        self.move_client  = actionlib.SimpleActionClient(GRIPPER_MOVE_SERVER,  MoveAction)
        self.grasp_client = actionlib.SimpleActionClient(GRIPPER_GRASP_SERVER, GraspAction)

        rospy.loginfo("Waiting for gripper action servers...")
        self.move_client.wait_for_server(rospy.Duration(5.0))
        self.grasp_client.wait_for_server(rospy.Duration(5.0))

        # ------------------------------------------------------------------ #
        #  Data collector service                                             #
        # ------------------------------------------------------------------ #
        try:
            rospy.wait_for_service('/data_collector/set_recording', timeout=5.0)
            self._svc_set_recording = rospy.ServiceProxy('/data_collector/set_recording', SetBool)
        except Exception:
            rospy.logwarn("Data collector service not found — replaying without recording.")
            self._svc_set_recording = None

        self.replay_rate = rospy.Rate(1.0 / DT)  # 10 Hz

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _move_to_start(self):
        """Use MoveIt planner for a safe, collision-aware move to start pose."""
        rospy.loginfo("Planning move to start pose with MoveIt...")
        self.arm.set_joint_value_target(self.joints_array[0].tolist())

        success, plan, _, _ = self.arm.plan()
        if not success:
            rospy.logerr("MoveIt could not plan to start pose!")
            sys.exit(1)

        rospy.loginfo("Executing move to start pose...")
        self.arm.execute(plan, wait=True)
        self.arm.stop()
        rospy.loginfo("At start pose.")

    def _build_trajectory(self):
        """
        Build a RobotTrajectory directly from recorded data.
        Positions and velocities come straight from hardware measurements.
        Accelerations are derived from the already-smooth hardware velocities.
        MoveIt's planner is NOT used here — this is pure replay.
        """
        rospy.loginfo("Building trajectory from recorded data...")

        trajectory = RobotTrajectory()
        jt = trajectory.joint_trajectory
        jt.joint_names = self.arm.get_active_joints()

        for i in range(self.n_steps):
            pt = JointTrajectoryPoint()
            pt.positions     = self.joints_array[i].tolist()
            pt.velocities    = self.velocities_array[i].tolist()   # from hardware
            pt.accelerations = self.accels_array[i].tolist()       # from np.gradient on hw vel
            pt.time_from_start = rospy.Duration.from_sec(i * DT)
            jt.points.append(pt)

        return trajectory

    def _send_gripper_step(self, step_idx, current_state):
        close_width = float(0.04)
        open_width = float(0.075)

        if current_state == 0 and (not self.gripper_state_closed):  # closing / grasping
            goal = GraspGoal(width=close_width, speed=GRIPPER_SPEED, force=GRIPPER_FORCE)
            #goal = GraspGoal(width=close_width)
            goal.epsilon.inner = goal.epsilon.outer = GRIPPER_EPSILON
            self.grasp_client.send_goal(goal)
            self.gripper_state_closed = True
            rospy.loginfo(f"[{step_idx}] Gripper: GRASPING  (width={close_width:.4f} m)")

        elif current_state ==1 and self.gripper_state_closed:                   # opening
            self.move_client.send_goal(MoveGoal(width=open_width,speed = GRIPPER_SPEED))
            self.gripper_state_closed = False
            rospy.loginfo(f"[{step_idx}] Gripper: OPENING   (width={open_width:.4f} m)")
        else:
            pass

    # ------------------------------------------------------------------ #
    #  Main                                                               #
    # ------------------------------------------------------------------ #

    def run(self):
        # 1. Safe move to start via MoveIt planner
        self._move_to_start()

        input("\nPress Enter to begin replay...\n")

        # 2. Start data collector
        

        # 3. Build trajectory from recorded data and execute via MoveIt
        trajectory = self._build_trajectory()
       
        if self._svc_set_recording:
            try:
                self._svc_set_recording(True)
            except Exception as e:
                rospy.logwarn(f"Could not start data collector: {e}")

        self.arm.execute(trajectory, wait=False)  
        
        # 4. Gripper sync loop
        last_state = -1
        for step_idx in range(self.n_steps):
            if rospy.is_shutdown():
                self.arm.stop()
                break

            current_state = int(self.gripper_state_array[step_idx])
            if current_state != last_state:
                self._send_gripper_step(step_idx, current_state)
                last_state = current_state

            if step_idx % 50 == 0 and step_idx > 0:
                rospy.loginfo(f"Progress: {100.0 * step_idx / self.n_steps:.1f}%")

            self.replay_rate.sleep()

        # 5. Wait for arm to finish
        rospy.loginfo("Gripper loop done. Waiting for arm to finish...")
        self.arm.stop()
        rospy.sleep(1.0)

        # 6. Save
        rospy.loginfo("Replay complete. Saving data...")
        if self._svc_set_recording:
            try:
                self._svc_set_recording(False)
            except Exception:
                pass

        moveit_commander.roscpp_shutdown()


# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 franka_replay_moveit.py <data.npz>")
        sys.exit(1)

    replayer = FrankaReplayerMoveIt(sys.argv[1])
    replayer.run()