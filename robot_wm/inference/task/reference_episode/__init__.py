from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ImageProprioGoal:
    image: np.ndarray  # (C, H, W) [0, 1]
    proprio: np.ndarray  # (D,) (x, y, z, qw, qx, qy, qz, gripper)

    def __post_init__(self):
        # Image should be in [0, 1], (C, H, W)
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

        # Proprio should be (7,)
        if self.proprio is not None:
            if not isinstance(self.proprio, np.ndarray):
                raise TypeError("proprio must be a numpy ndarray")
            if len(self.proprio.shape) != 1:
                raise ValueError(f"proprio must be a 1D array, got shape {self.proprio.shape}")
            if self.proprio.shape[0] != 7:
                raise ValueError(f"proprio must have 7 elements (x,y,z,euler angles,gripper), got {self.proprio.shape[0]}")

    def subsample_subgoal(self, start_time, end_time):
        return self
    
    def ZERO():
        C, H, W = 3, 224, 224
        return ImageProprioGoal(np.zeros((C, H, W)), np.array([0, 0, 0, 1, 0, 0, 0, 0]))

    def show(self):
        import matplotlib.pyplot as plt

        plt.title("Goal Image")
        plt.imshow(self.image.transpose(1, 2, 0))
        plt.show(block=False)

    def write_image_to_file(self, png_path):
        import os
        from matplotlib import pyplot as plt
        os.makedirs(os.path.dirname(png_path), exist_ok=True)
        img = self.image.transpose(1, 2, 0)
        plt.imsave(png_path, img)

    @property
    def last_image(self):
        return self.image
    
    @property
    def image_tensor(self):
        return torch.tensor(self.image).unsqueeze(0)  # (1, 3, H, W)

    @property
    def proprio_tensor(self):
        return torch.tensor(self.proprio).unsqueeze(0)  # (1, 7)
    
@dataclass
class VisualDemonstrationGoal:
    image: np.ndarray  # (T, C, H, W) [0, 1]
    proprio: np.ndarray  # (T, D) (x, y, z, qw, qx, qy, qz, gripper)
    freq: int = 30  # Hz

    def __post_init__(self):
        # Image should be in [0, 1], (T, C, H, W)
        if not isinstance(self.image, np.ndarray):
            raise TypeError("image must be a numpy ndarray")
        if len(self.image.shape) != 4:
            raise ValueError(
                f"image must have 4 dimensions (T, C, H, W), got shape {self.image.shape}"
            )
        T, C, H, W = self.image.shape
        if C != 3:
            raise ValueError(
                f"image must have 3 channels and be channel first (T, C, H, W), got {self.image.shape}"
            )
        if np.any(self.image < 0) or np.any(self.image > 1):
            raise ValueError("image values must be between 0 and 1")

        # Proprio should be (7,)
        if not isinstance(self.proprio, np.ndarray):
            raise TypeError("proprio must be a numpy ndarray")
        if len(self.proprio.shape) != 2:
            raise ValueError(f"proprio must be a (T, 7) array, got shape {self.proprio.shape}")
        T, D = self.proprio.shape
        if D != 7:
            raise ValueError(f"proprio must have 7 elements (x,y,z,euler angles,gripper), got {self.proprio.shape}")

    def subsample_subgoal(self, start_time, end_time):
        start_idx = int(start_time * self.freq)
        start_idx = min(len(self.image) - 1, start_idx) # guarantee at least 1 frame
        end_idx = int(end_time * self.freq)
        end_idx = min(len(self.image), end_idx) # guarantee at least 1 frame
        new_image = self.image[start_idx:end_idx]
        new_proprio = self.proprio[start_idx:end_idx]
        new_freq = self.freq
        return VisualDemonstrationGoal(new_image, new_proprio, new_freq)

    @property
    def last_image(self):
        return self.image[-1]
    
    def ZERO():
        T, C, H, W = 1, 3, 224, 224
        return VisualDemonstrationGoal(np.zeros((T, C, H, W)), np.array([[0, 0, 0, 1, 0, 0, 0, 0]]))

    def show(self):
        
        import matplotlib.pyplot as plt

        plt.title("Goal Video")
        # sample 5 frames equally spaced
        T, C, H, W = self.image.shape
        N = min(5, T)
        frames_i = np.linspace(0, T - 1, N).astype(int)
        fig, axs = plt.subplots(1, N, figsize=(N * 3, 3))
        for i, frame_i in enumerate(frames_i):
            axs[i].imshow(self.image[frame_i].transpose(1, 2, 0))
            axs[i].set_title(f"Frame {frame_i}")
            axs[i].axis("off")
        plt.show(block=False)

    def write_image_to_file(self, png_path):
        import os
        from matplotlib import pyplot as plt
        # sample 5 frames equally spaced
        T, C, H, W = self.image.shape
        N = min(5, T)
        frames_i = np.linspace(0, T - 1, N).astype(int)
        frames = self.image[frames_i]
        concat_image = np.concatenate(frames, axis=2)  # (C, H, N*W)
        os.makedirs(os.path.dirname(png_path), exist_ok=True)
        img = concat_image.transpose(1, 2, 0)
        plt.imsave(png_path, img)

    @property
    def image_tensor(self):
        return torch.tensor(self.image)

    @property
    def proprio_tensor(self):
        return torch.tensor(self.proprio)

class ReferenceEpisode:
    pass
