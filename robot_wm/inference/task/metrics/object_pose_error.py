import apriltag
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arrow

from robot_wm.inference.robot.base import RobotAction, RobotObs
from robot_wm.inference.task.metrics import Metric
from robot_wm.inference.task.reference_episode.h5_imagegoal_reference_episode import (
    ImageProprioGoal,
)


def detect_apriltags(image_path, families="tag36h11"):
    """Detect AprilTags in an image and visualize their position and orientation.

    Args:
        image_path (str): Path to the input image
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Initialize AprilTag detector
    # Using pupil_apriltags which is a Python wrapper for the C implementation
    options = apriltag.DetectorOptions(families=families)
    detector = apriltag.Detector(options)

    # Detect AprilTags
    results = detector.detect(gray)
    print(f"Detected {len(results)} AprilTags")

    # Create RGB version of image for display
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create figure and axis for plotting
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(rgb_img)

    # Process each tag
    for r in results:
        # Extract tag information
        tag_id = r.tag_id
        center = r.center
        corners = r.corners

        # Print information
        print(f"Tag ID: {tag_id}")
        print(f"Center: ({center[0]:.2f}, {center[1]:.2f})")

        # Draw tag outline
        for i in range(4):
            j = (i + 1) % 4  # Wrap around the corner indices
            ax.plot(
                [corners[i][0], corners[j][0]],
                [corners[i][1], corners[j][1]],
                "g-",
                linewidth=2,
            )

        # Indicate tag ID
        ax.text(
            center[0],
            center[1],
            str(tag_id),
            fontsize=12,
            color="red",
            bbox=dict(facecolor="white", alpha=0.7),
        )

        # Draw orientation vector (from center to first corner)
        direction = (
            corners[0] - center
        )  # Use the first corner as reference for orientation
        direction_norm = (
            direction / np.linalg.norm(direction) * 20
        )  # Normalize and scale for better visualization

        # Draw arrow using matplotlib's Arrow patch
        arrow = Arrow(
            center[0],
            center[1],
            direction_norm[0],
            direction_norm[1],
            width=10,
            color="blue",
        )
        ax.add_patch(arrow)

    # Set plot title and hide axis
    ax.set_title("AprilTag Detection")
    ax.axis("off")

    # Tight layout for better spacing
    plt.tight_layout()

    # Save the figure and display the plot
    plt.savefig("output_image.png")
    plt.show()

    return results


def get_center_apriltags(img, tag_id_of_obj, families="tag36h11"):
    """Detect AprilTags in an image and visualize their position and orientation.

    Args:
        img is np array of the image
        tag_id_of_obj is the id of the AprilTag to be detected which is pasted on the object
        families is the family of AprilTag to be detected
    """
    # Initialize AprilTag detector
    # Using pupil_apriltags which is a Python wrapper for the C implementation
    options = apriltag.DetectorOptions(families=families)
    detector = apriltag.Detector(options)

    transposed_array = np.transpose(img, (1, 2, 0))
    transposed_array = (transposed_array * 255).astype(np.uint8)
    gray = cv2.cvtColor(transposed_array, cv2.COLOR_BGR2GRAY)

    # Detect AprilTags
    results = detector.detect(gray)

    # Process each tag
    for r in results:
        if r.tag_id == tag_id_of_obj:
            return r.center


class ObjectPoseErrorMetric(Metric):
    """
    This metric computes the error between the goal and the reached object pose.
    The error is computed as the sum of the absolute differences between the center of the AprilTag in Pixel space of Realsense img
    currently ignoring corners, homography and not getting the 3D pose of the object
    id_of_april_tag is the id of the AprilTag to be detected which is pasted on the object
    id_of_april_tag needs to be the same for goal img and obs img
    """

    def __init__(self, id_of_april_tag):
        self.id_of_april_tag = id_of_april_tag

    def reset(self):
        pass

    def compute(self, obs: RobotObs, action: RobotAction, goal: ImageProprioGoal):
        goal_img = goal.image
        obs_img = obs.image
        goal_center = get_center_apriltags(goal_img, self.id_of_april_tag)
        obs_center = get_center_apriltags(obs_img, self.id_of_april_tag)

        if goal_center is None or obs_center is None:
            print(
                "Error in ObjectPoseErrorMetric: Could not detect AprilTag in one of the images."
            )
            return None, {"error": None, "goal_center": None, "obs_center": None}
        # Compute the position difference as the sum of absolute differences
        # between the goal and reached positions (in pixel coordinates)
        error_array = np.array(goal_center - obs_center)
        error = np.sum(np.abs(error_array))
        return (
            error,
            {"dist_apriltag_center": error},
        )
