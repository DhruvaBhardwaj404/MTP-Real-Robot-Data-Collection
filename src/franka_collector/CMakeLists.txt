cmake_minimum_required(VERSION 3.0.2)
project(franka_collector)

find_package(catkin REQUIRED COMPONENTS
  rospy
  roscpp
  sensor_msgs
  cv_bridge
  std_srvs
  franka_gripper
  control_msgs
  trajectory_msgs
  moveit_ros_planning_interface
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/franka_data_collector.py
  scripts/franka_replay_moveit.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)