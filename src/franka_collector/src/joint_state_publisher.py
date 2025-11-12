#!/usr/bin/env python3

import rospy
import math
from sensor_msgs.msg import JointState


def joint_state_publisher():
    """
    Publishes synthetic JointState data for a 7-DOF robot (like Franka Emika Panda)
    on the topic '/franka/joint_states'.
    This is just for testing without the robot
    """
    # Initialize the ROS node
    rospy.init_node('dummy_joint_state_publisher', anonymous=True)

    # Create a publisher for the JointState topic
    pub = rospy.Publisher('/franka/joint_states', JointState, queue_size=10)

    # Define the rate at which to publish (e.g., 50 Hz)
    rate = rospy.Rate(50)

    # Define joint names (typical for a 7-DOF arm plus gripper)
    joint_names = [
        'panda_joint1', 'panda_joint2', 'panda_joint3',
        'panda_joint4', 'panda_joint5', 'panda_joint6',
        'panda_joint7', 'panda_finger_joint1'
    ]

    # Counter for simulating movement
    start_time = rospy.Time.now().to_sec()

    rospy.loginfo("Dummy Joint State Publisher started and publishing to /franka/joint_states.")

    while not rospy.is_shutdown():
        # Get current time for sinusoidal movement calculation
        current_time = rospy.Time.now().to_sec()
        time_elapsed = current_time - start_time

        # Initialize the JointState message
        joint_state_msg = JointState()
        joint_state_msg.header.stamp = rospy.Time.now()
        joint_state_msg.name = joint_names

        # Generate synthetic data (slow sinusoidal movement)
        positions = []
        velocities = []

        for i in range(len(joint_names)):
            # Positions vary sinusoidally
            pos = 0.5 * math.sin(time_elapsed / (i + 1) * 0.5)
            positions.append(pos)

            # Velocities are the derivative of position
            vel = 0.25 * math.cos(time_elapsed / (i + 1) * 0.5)
            velocities.append(vel)

        joint_state_msg.position = positions
        joint_state_msg.velocity = velocities

        # Publish the message
        pub.publish(joint_state_msg)

        # Sleep to maintain the desired rate
        rate.sleep()


if __name__ == '__main__':
    try:
        joint_state_publisher()
    except rospy.ROSInterruptException:
        pass