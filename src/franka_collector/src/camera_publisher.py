#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


def camera_publisher():
    """
    Publishes a dummy grayscale image (640x480) on the topic '/camera/image_raw'.
    This is for testing without the robot
    """
    # Initialize the ROS node
    rospy.init_node('dummy_camera_publisher', anonymous=True)

    # Create a publisher for the Image topic
    pub = rospy.Publisher('/camera/image_raw', Image, queue_size=1)

    # Initialize CvBridge for converting OpenCV images to ROS messages
    bridge = CvBridge()

    # Define the publishing rate (e.g., 10 Hz)
    rate = rospy.Rate(10)

    # Create a dummy image (640x480 grayscale, value 128)
    width, height = 640, 480
    dummy_image = np.full((height, width), 128, dtype=np.uint8)

    rospy.loginfo("Dummy Camera Publisher started and publishing to /camera/image_raw.")

    while not rospy.is_shutdown():
        # Add a dynamic element to the image (e.g., a simple counter or line)
        current_time = int(rospy.Time.now().to_sec() * 10) % height
        image_copy = dummy_image.copy()
        cv2.line(image_copy, (0, current_time), (width - 1, current_time), 255, 5)

        try:
            # Convert the OpenCV image to a ROS Image message
            ros_image_msg = bridge.cv2_to_imgmsg(image_copy, encoding="mono8")
            ros_image_msg.header.stamp = rospy.Time.now()
            ros_image_msg.header.frame_id = "camera_link"

            # Publish the message
            pub.publish(ros_image_msg)

        except CvBridge.CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

        # Sleep to maintain the desired rate
        rate.sleep()


if __name__ == '__main__':
    try:
        camera_publisher()
    except rospy.ROSInterruptException:
        pass