import threading
import time
from collections import deque

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32MultiArray

from robot_wm.inference.common.robot_obs_logger import DataLoggerFranka
from robot_wm.inference.robot.base import RobotEEDeltaAction, RobotObs, RobotState


class FrankaRobot(Node):
    def __init__(self, history_size=1):
        # Initialize rclpy if not already initialized
        if not rclpy.ok():
            rclpy.init()
        super().__init__("franka_agent")

        # Create reliable QoS profile
        self.qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        # Setup state history containers
        self.history_size = history_size
        self.rgb_images = deque(maxlen=history_size)
        self.joint_states = deque(maxlen=history_size)
        self.eef_poses = deque(maxlen=history_size)
        self.data_storage_freq = 10  # in hz #TODO: make this a parameter
        # Setup CV bridge for image conversion
        self.cv_bridge = CvBridge()

        # Setup publishers and subscribers
        self.setup_subscribers()
        self.setup_publishers()

        # Threading variables
        self.spin_thread = None
        self.is_spinning = False
        self.spin_lock = threading.Lock()
        # Logger
        self.logger = DataLoggerFranka()

    def start(self):
        """Start spinning the node in a separate thread."""
        with self.spin_lock:
            if self.is_spinning:
                self.get_logger().info("Node is already spinning")
                return

            self.is_spinning = True
            self.spin_thread = threading.Thread(target=self._spin_thread_func)
            self.spin_thread.daemon = True  # Thread will exit when main program exits
            self.spin_thread.start()
            self.get_logger().info("Started spinning in a separate thread")
            self.store_obs_thread = threading.Thread(target=self._store_obs_thread)

    def stop(self):
        """Stop the spinning thread and clean up ROS resources."""
        with self.spin_lock:
            if not self.is_spinning:
                return

            self.is_spinning = False
            if self.spin_thread:
                # Give the thread time to exit cleanly
                self.spin_thread.join(timeout=1.0)
                self.get_logger().info("Stopped spinning thread")

    def _store_obs_thread(self):
        while self.is_spinning:
            obs = self.get_last_observation()
            if obs is not None:
                self.logger.log_obs_franka(obs)
            time.sleep(1 / self.data_storage_freq)

    def _spin_thread_func(self):
        """Function that runs in the spin thread."""
        try:
            # Keep a reference to the node to be spun
            executor = rclpy.executors.SingleThreadedExecutor()
            executor.add_node(self)

            # Use spin instead of spin_once for continuous spinning
            while self.is_spinning and rclpy.ok():
                try:
                    executor.spin_once(timeout_sec=0.1)
                except Exception as inner_e:
                    # Handle temporary errors that might occur during spinning
                    self.get_logger().warn(f"Temporary error in spin: {inner_e}")
                    time.sleep(0.1)  # Add a small delay before retrying
        except Exception as e:
            self.get_logger().error(f"Error in spin thread: {e}")
        finally:
            # Clean up executor if needed
            if "executor" in locals():
                executor.remove_node(self)

            self.get_logger().info("Spin thread exiting")

    def execute(self, action: RobotEEDeltaAction):
        return self.execute_delta_action(action)

    def get_last_observation(self):
        if (
            len(self.joint_states) == 0
            or len(self.eef_poses) == 0
            or len(self.rgb_images) == 0
        ):
            return None

        image_HWC_255 = np.array(self.rgb_images[-1])
        image_CHW_01 = image_HWC_255.transpose(2, 0, 1) / 255.0
        image_CHW_01 = image_CHW_01[[2, 1, 0], :, :]

        return RobotObs(
            joints=np.array(self.joint_states[-1]),
            end_effector_pose=np.array(self.eef_poses[-1]),
            image=image_CHW_01,
        )

    def setup_subscribers(self):
        """Setup all the necessary subscribers."""
        self.joint_state_sub = self.create_subscription(
            JointState,
            "/franka/joint_states",
            self.joint_state_callback,
            self.qos_profile,
        )
        self.eef_pose_sub = self.create_subscription(
            PoseStamped, "/franka/eef_pose", self.eef_pose_callback, self.qos_profile
        )
        self.rgb_image_sub = self.create_subscription(
            Image,
            "/right_camera/color/image_raw",
            self.rgb_image_callback,
            self.qos_profile,
        )

    def joint_state_callback(self, msg):
        """Process incoming joint state messages."""
        self.joint_states.append(msg.position)

    def eef_pose_callback(self, msg):
        """Process incoming end effector pose messages."""
        self.eef_poses.append(
            np.array(
                [
                    msg.pose.position.x,
                    msg.pose.position.y,
                    msg.pose.position.z,
                    msg.pose.orientation.w,
                    msg.pose.orientation.x,
                    msg.pose.orientation.y,
                    msg.pose.orientation.z,
                ]
            )
        )

    def rgb_image_callback(self, msg):
        """Process incoming RGB image messages."""
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.rgb_images.append(cv_image)
        except Exception as e:
            self.get_logger().error(f"Error converting RGB image: {e}")

    def setup_publishers(self):
        self.delta_pose_pub = self.create_publisher(
            Float32MultiArray, "/franka/target_delta_pose_cmd", self.qos_profile
        )
        self.reset_joints_pub = self.create_publisher(
            Float32MultiArray, "/franka/reset_joints_cmd", self.qos_profile
        )

    def reset_to_state(self, state: RobotState, timeout=10.0) -> RobotObs:
        msg = Float32MultiArray()
        msg.data = state.joints.tolist()
        self.reset_joints_pub.publish(msg)
        self.get_logger().info(f"Reset joints to state: {state}")
        # move until reached or timeout
        start_time = time.time()
        JOINT_TOL = 0.1
        while True:
            # If not spinning in a thread, spin manually
            if not self.is_spinning:
                rclpy.spin_once(self, timeout_sec=0.1)

            obs = self.get_last_observation()
            if time.time() - start_time > timeout:
                if obs is None:
                    self.get_logger().warn(
                        f"Still no observations, joint states: {len(self.joint_states)}, eef poses: {len(self.eef_poses)}, rgb images: {len(self.rgb_images)}"
                    )
                self.get_logger().warn("Reset joints timed out")
                raise TimeoutError("Reset joints timed out")
                break
            if obs is None:
                continue
            if np.linalg.norm(np.array(obs.joints) - state.joints) < JOINT_TOL:
                break
            # Add a small sleep to avoid busy waiting
            time.sleep(0.01)
        return self.get_last_observation()

    def execute_delta_action(self, delta_action: RobotEEDeltaAction):
        msg = Float32MultiArray()
        SAFE = 1.0
        arm_action = delta_action.action[:6] * SAFE
        # gripper_action = delta_action.action[-1]  # TODO
        vel_mms = 30
        msg.data = [float(a) for a in arm_action[0]] + [float(vel_mms)]
        self.delta_pose_pub.publish(msg)
        self.get_logger().info(f"Executing action: {delta_action}")

        # Only spin if not already spinning in a thread
        if not self.is_spinning:
            # SPIN FOR A 10MS
            start_time = time.time()
            while time.time() - start_time < 0.01:
                rclpy.spin_once(self)

        return self.get_last_observation(), {}

    def __del__(self):
        """Destructor to ensure clean shutdown."""
        try:
            self.stop()
        except Exception as e:
            # Can't use get_logger in __del__
            print(f"Error during FrankaRobot cleanup: {e}")
