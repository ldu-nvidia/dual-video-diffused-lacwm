from dataclasses import dataclass

import numpy as np


@dataclass
class Robot:
    pass


@dataclass
class RobotState:
    joints: np.ndarray  # (7,)

    def __post_init__(self):
        # Validate joints is a numpy ndarray with shape (7,)
        if not isinstance(self.joints, np.ndarray):
            raise TypeError("joints must be a numpy ndarray")
        if self.joints.shape != (7,):
            raise ValueError(f"joints must have shape (7,), got {self.joints.shape}")

    def ZERO():
        return RobotState(np.zeros(7))


@dataclass
class RobotObs:
    joints: np.ndarray  # (7,)
    end_effector_pose: np.ndarray  # (7,) x, y, z, r, p, y, gripper
    image: np.ndarray  # (3, 224, 224) RGB between 0 and 1

    def __post_init__(self):
        # Validate joints is a numpy ndarray with shape (7,)
        if not isinstance(self.joints, np.ndarray):
            raise TypeError("joints must be a numpy ndarray")
        if self.joints.shape != (7,):
            raise ValueError(f"joints must have shape (7,), got {self.joints.shape}")

        # Validate end_effector_pose is a numpy ndarray with shape (7,)
        if not isinstance(self.end_effector_pose, np.ndarray):
            raise TypeError("end_effector_pose must be a numpy ndarray")
        if self.end_effector_pose.shape != (7,):
            raise ValueError(
                f"end_effector_pose must have shape (7,), got {self.end_effector_pose.shape}"
            )

        # Validate image is a numpy ndarray with shape (3, 224, 224) and values between 0 and 1
        if not isinstance(self.image, np.ndarray):
            raise TypeError("image must be a numpy ndarray")
        if len(self.image.shape) != 3:
            raise ValueError(
                f"image must have 3 dimensions (C, H, W), got shape {self.image.shape}"
            )
        if self.image.shape[0] != 3:
            raise ValueError(
                f"image must have 3 channels and be channel first (C, H, W), got {self.image.shape[0]}"
            )
        if np.any(self.image < 0) or np.any(self.image > 1):
            raise ValueError("image values must be between 0 and 1")

    def ZERO():
        return RobotObs(np.zeros(7), np.zeros(7), np.zeros((3, 224, 224)))

    def show(self):
        import matplotlib.pyplot as plt

        plt.title("Robot Image")
        plt.imshow(self.image.transpose(1, 2, 0))
        plt.show(block=False)


@dataclass
class RobotAction:
    action: np.ndarray  # batch_size x action_dim

    def __post_init__(self):
        # Validate action is a numpy ndarray
        if not isinstance(self.action, np.ndarray):
            raise TypeError("action must be a numpy ndarray")

        # Make sure it has 2 dimensions, if only 1, add a dimension
        if len(self.action.shape) == 1:
            self.action = self.action[None, :]

        if len(self.action.shape) != 2:
            raise ValueError("action must have 2 dimensions, batch_size x action_dim")

    def __add__(self, other):
        """Define how to add two RobotAction objects together."""
        if isinstance(other, RobotAction):
            return RobotAction(action=self.action + other.action)
        return NotImplemented

    def __radd__(self, other):
        """Support for sum() which starts with 0 + object."""
        if other == 0:
            return self
        return self.__add__(other)

    @property
    def action_dim(self):
        return self.action.shape[1]


@dataclass
class RobotEEDeltaAction(RobotAction):
    action: np.ndarray  # batch_size x action_dim -- (1,7) dx, dy, dz, dr, dp, dy, dgripper

    def __post_init__(self):
        # Validate action is a numpy ndarray
        if not isinstance(self.action, np.ndarray):
            raise TypeError("action must be a numpy ndarray")

        # Make sure it has 1 dimension, if only 1, add a dimension
        if len(self.action.shape) == 1:
            self.action = self.action[None, :]

        # Validate action has shape (7,)
        if self.action.shape[1] != 7:
            raise ValueError(f"EEDelta action should have 7, got {self.action.shape}")

    def __add__(self, other):
        """Define how to add two RobotEEDeltaAction objects together."""
        if isinstance(other, RobotEEDeltaAction):
            new_action = np.copy(self.action)

            # Add translation (x, y, z)
            new_action[:3] += other.action[:3]

            # Add rotation (dr, dp, dy) -- wrap angles
            new_action[3:6] += other.action[3:6]
            new_action[3:6] = np.vectorize(self.wrap_angle)(new_action[3:6])

            # Average gripper actions
            new_action[6] = (self.action[6] + other.action[6]) / 2

            return RobotEEDeltaAction(new_action)

        return NotImplemented

    @property
    def action_dim(self):
        return self.action.shape[1]

    def is_zero_action(self):
        return np.allclose(self.action, 0)

    def wrap_angle(self, angle):
        """Wrap an angle to the range [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def batch_tile(self, B):
        """Tile the action to have batch size B."""
        if self.action.shape[0] == B:
            return self
        elif self.action.shape[0] == 1:
            return RobotEEDeltaAction(np.tile(self.action, (B, 1)))
        else:
            raise ValueError(f"Action shape {self.action.shape} must have batch size 1 or {B} to tile")


    def ZERO():
        return RobotEEDeltaAction(np.zeros(7))

    def RANDOM():
        # Small enough to be safe
        return RobotEEDeltaAction(np.random.uniform(-0.05, 0.05, 7))
