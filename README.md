rosrun franka_collector data_collector_controlled.py


To start Recording:
rostopic pub -1 /record/start std_msgs/Bool "data: true"
To end Recording:
rostopic pub -1 /record/start std_msgs/Bool "data: false"
