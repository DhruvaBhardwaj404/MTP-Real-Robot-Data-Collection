# --- eval_franka_ros_policy.py ---

import time
import click
import numpy as np
import torch
import dill
import hydra
import pathlib
from omegaconf import OmegaConf

# ROS Imports
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray

# Policy Imports (unchanged)
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply

OmegaConf.register_new_resolver("eval", eval, replace=True)

# Global variables to store the latest camera and robot state
# These will be updated by ROS subscribers
current_ros_obs = {}
image_transformers = {}  # Store CvBridge and image processing logic


# --- ROS Callback Functions ---

def robot_state_callback(msg: Float32MultiArray):
    """Callback for receiving the 6D robot end-effector pose (x, y, z, Rx, Ry, Rz)."""
    # Assuming the ROS node publishes a Float32MultiArray with 6 elements
    # and the policy expects a key like 'robot_eef_pose'.
    global current_ros_obs
    pose_6d = np.array(msg.data, dtype=np.float64)
    current_ros_obs['robot_eef_pose'] = pose_6d
    # Note: ROS time should be converted to match your policy's expected time format if necessary


def camera_callback(msg: Image, camera_id: str):
    """Callback for receiving compressed camera images from ROS."""
    global current_ros_obs, image_transformers

    # You will need a CvBridge instance (and possibly `get_image_transform`)
    # to convert ROS Image message to the NumPy array/Tensor your policy expects.
    # The image must be processed to match the resolution/format used for training.

    # Example:
    # cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    # transformed_image = image_transformers[camera_id](cv_image)
    # current_ros_obs[camera_id] = transformed_image

    # Placeholder for actual transformation
    current_ros_obs[camera_id] = np.zeros((128, 128, 3), dtype=np.float32)


@click.command()
# ... (click options remain the same)
def main(input, output, robot_ip, match_dataset, match_episode,
         vis_camera_idx, init_joints,
         steps_per_inference, max_duration,
         frequency, command_latency):
    # Initialize ROS Node
    rospy.init_node('policy_inference_node', anonymous=True)

    # --- Policy Initialization (remains largely the same) ---
    ckpt_path = input
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # ... (Policy type-specific hacks: diffusion, robomimic, ibc)
    # ... (Set policy, device, steps_per_inference, etc.)

    # Get observation metadata from the loaded configuration
    shape_meta = cfg.task.shape_meta

    # --- ROS Subscriptions and Publishers ---

    # Publisher for policy actions (e.g., target 6D pose)
    action_pub = rospy.Publisher('/franka_policy/target_pose', PoseStamped, queue_size=1)

    # Subscribers for robot state (must match your ROS setup)
    rospy.Subscriber('/franka_policy/robot_state', Float32MultiArray, robot_state_callback)

    # Subscribers for cameras (must match your ROS setup and cfg.task.shape_meta)
    camera_names = [k for k in shape_meta['obs']['keys'] if k.startswith('camera_')]
    for cam_name in camera_names:
        # Assuming ROS camera topics are named like /camera_0/image_raw
        rospy.Subscriber(f'/{cam_name.replace("_", "/")}/image_raw',
                         Image,
                         lambda msg, cam=cam_name: camera_callback(msg, cam),
                         queue_size=1)
        # Initialize image transformer here if using CvBridge

    print('ROS Subscribers and Publisher set up. Waiting for first observation...')

    # Wait until all required observations are available
    while not all(key in current_ros_obs for key in shape_meta['obs']['keys']) and not rospy.is_shutdown():
        rospy.sleep(0.1)

    # --- Policy Control Loop ---

    rospy.loginfo("Policy inference starting.")

    rate = rospy.Rate(frequency)
    policy.reset()  # Reset policy state (e.g., for recurrent models)

    while not rospy.is_shutdown():
        # 1. Get current observation (from global ROS-updated dictionary)
        # Convert dictionary to batch format for policy
        obs_dict = current_ros_obs.copy()  # Use a copy to prevent race conditions

        # NOTE: Your policy expects N_OBS_STEPS history.
        # In a real ROS setup, you would need a small buffer to store N_OBS_STEPS
        # of data and then format it here. Since ROS callbacks only provide the
        # *latest* data, this is a major simplification.

        # Simplified: Use latest observation and repeat for N_OBS_STEPS
        # You MUST implement a proper buffer management system for multi-step history.
        input_obs = {}
        for k, v in obs_dict.items():
            # Stack N_OBS_STEPS of data: (N_OBS_STEPS, H, W, C) or (N_OBS_STEPS, D)
            input_obs[k] = np.stack([v] * cfg.n_obs_steps, axis=0)

            # Convert to Pytorch tensor and batch it
        obs_tensor = dict_apply(input_obs,
                                lambda x: torch.from_numpy(x).to(device).unsqueeze(0))

        # 2. Run Policy Inference
        with torch.no_grad():
            action_preds = policy.predict_action(obs_tensor)

        # Extract the action for the first step
        # (B, H, D_action) -> (D_action)
        action_pred = action_preds[0, action_offset].detach().cpu().numpy()

        # 3. Publish Action to ROS

        # Create a ROS PoseStamped message from the predicted 6D pose (action_pred)
        # Your ROS receiver node must handle the conversion from (Rx, Ry, Rz) to a Rotation matrix.
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = "world"  # Or your robot's base frame

        pose_msg.pose.position.x = action_pred[0]
        pose_msg.pose.position.y = action_pred[1]
        pose_msg.pose.position.z = action_pred[2]

        # Assuming action_pred[3:] are (Rx, Ry, Rz) rotation vector components.
        # It's better to convert them to quaternions for standard PoseStamped messages.
        rot_vec = action_pred[3:]
        rotation = st.Rotation.from_rotvec(rot_vec)
        quat = rotation.as_quat()  # [x, y, z, w]

        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]

        action_pub.publish(pose_msg)

        # 4. Control Loop Frequency
        rate.sleep()


if __name__ == '__main__':
    main()