from __future__ import annotations

from base64 import b64encode
from typing import Optional, Sequence

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import HTML, display
from PIL import Image


def prepare_inputs(
    episode: dict[str, torch.Tensor],
    start_frame_idx: int,
    n_context_frames: int,
    clear_actions: bool = False,
) -> dict[str, torch.Tensor]:
    for key in episode:
        episode[key] = episode[key].to("cuda").to(torch.bfloat16)

    rgb = episode["rgb"].detach().clone().unsqueeze(0)
    actions = episode["actions"].detach().clone().unsqueeze(0)

    if clear_actions:
        print("WARNING: clearing actions")
        actions[:, start_frame_idx - 1 :] = 0.0

    context_frame_idx = start_frame_idx - n_context_frames
    assert context_frame_idx >= 0

    return {
        "rgb": rgb[:, context_frame_idx:start_frame_idx],
        "actions": actions[:, context_frame_idx:],
    }


def _to_image(
    x: torch.Tensor,
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
    add_border: bool = False,
) -> torch.Tensor:
    x = x.permute(1, 2, 0).float().to("cpu").numpy()
    if mean is not None:
        x = (x * np.array(std)) + np.array(mean)
    x = 255 * x
    x = np.uint8(x)
    if add_border:
        x[:, :5] = [255, 0, 0]
        x[:, -5:] = [255, 0, 0]
        x[:5, :] = [255, 0, 0]
        x[-5:, :] = [255, 0, 0]
    return x


def view_rollout(
    rgb: torch.Tensor,
    rgb_hat: Optional[torch.Tensor] = None,
    start_frame_idx: int = 1,
    n_context_frames: int = 1,
    skip: int = 1,
):
    rgb = rgb.squeeze(0)
    if rgb_hat is not None:
        rgb_hat = rgb_hat.squeeze(0)

    context_frame_idx = start_frame_idx - n_context_frames
    assert context_frame_idx >= 0

    view_indices = list(range(0, len(rgb), skip))

    ncols = len(view_indices)
    nrows = 1 if rgb_hat is None else 2
    _, axs = plt.subplots(nrows, ncols, squeeze=False, figsize=(ncols * 3, 3))
    for ax, idx in zip(axs[0], view_indices):
        ax.axis("off")
        ax.set_title(f"{idx}")
        ax.imshow(_to_image(rgb[idx]))
    if rgb_hat is not None:
        for ax, idx in zip(axs[1], view_indices):
            ax.axis("off")
            if idx == 0:
                ax.imshow(_to_image(rgb[idx]))
            else:
                add_border = idx >= start_frame_idx - context_frame_idx
                ax.imshow(_to_image(rgb_hat[idx - 1], add_border=add_border))


def save_rollout(
    rgb: torch.Tensor,
    rgb_hat: torch.Tensor,
    start_frame_idx: int,
    n_context_frames: int,
    output_path: str = "output.gif",
):
    rgb = rgb.squeeze(0)
    rgb_hat = rgb_hat.squeeze(0)

    context_frame_idx = start_frame_idx - n_context_frames
    assert context_frame_idx >= 0

    images = []
    pad = np.zeros((rgb.shape[-2], 10, 3))
    for idx in range(len(rgb)):
        left = _to_image(rgb[idx])
        if idx == 0:
            right = left.copy()
        else:
            add_border = idx >= start_frame_idx - context_frame_idx
            right = _to_image(rgb_hat[idx - 1], add_border=add_border)
        img = np.concatenate([left, pad, right], axis=1)
        images.append(Image.fromarray(img.astype(np.uint8)))

    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=200,
        loop=0,
    )


def create_video_from_frames(
    frames, fps=30, filepath="output_video.mp4", text_info_fn=None
):
    """
    Create and save a video from a sequence of frames using imageio.
    Parameters:
    -----------
    frames : list of numpy arrays
        List of image frames to create video from
    fps : int
        Frames per second
    filepath : str
        Output video file path
    text_info_fn : callable, optional
        Function that takes frame index and returns text to display
    """
    if isinstance(frames, torch.Tensor):
        frames = frames.cpu().numpy()
    if isinstance(frames, np.ndarray):
        frames = [frames[i] for i in range(frames.shape[0])]
    if not frames:
        print("No frames to save.")
        return None

    processed_frames = []
    for i, frame in enumerate(frames):
        # Make a copy to avoid modifying original
        frame = np.array(frame).copy()

        # Convert from (C,H,W) to (H,W,C) if needed
        if frame.shape[0] == 3 and len(frame.shape) == 3:
            frame = frame.transpose(1, 2, 0)

        # Convert to uint8 if needed
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)

        # Ensure the array is contiguous
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame)

        # Add text if provided
        if text_info_fn:
            try:
                text = text_info_fn(i)
                font = cv2.FONT_HERSHEY_SIMPLEX
                y_pos = 20
                for line in text.split("\n"):
                    cv2.putText(
                        frame,
                        line,
                        (10, y_pos),
                        font,
                        0.3,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    y_pos += 20
            except Exception as e:
                print(f"Warning: Could not add text to frame {i}: {e}")

        processed_frames.append(frame)

    # Save video
    imageio.mimsave(filepath, processed_frames, fps=fps)
    import os
    print(f"Video saved to {os.path.abspath(filepath)}")
    return filepath


def display_video_in_notebook(video_path):
    """
    Display a video in a Jupyter notebook.
    Parameters:
    -----------
    video_path : str
        Path to the video file
    """
    try:
        with open(video_path, "rb") as f:
            video_data = f.read()
            data_url = "data:video/mp4;base64," + b64encode(video_data).decode()
            # Fix: Use curly braces for string formatting instead of %
            html = HTML(
                f"""
            <video alt="video" controls style="max-width: 100%;">
              <source src="{data_url}" type="video/mp4">
            </video>
            """
            )
            display(html)
        return html
    except Exception as e:
        print(f"Error displaying video: {e}")
        return None
