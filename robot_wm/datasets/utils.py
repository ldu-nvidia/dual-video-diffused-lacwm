import json
import math
from typing import Literal

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def axisangle2quat(vecs):
    """
    Batched axis-angle (exponential coord) to quaternion.

    Args:
        vecs: np.array of shape (B, 3).

    Returns:
        np.array of shape (B, 4) where each row is (x, y, z, w).
    """
    vecs = np.asarray(vecs)
    angles = np.linalg.norm(vecs, axis=1)  # (B,)

    # Prepare output
    quats = np.zeros((vecs.shape[0], 4), dtype=float)  # (B, 4)

    # Handle non-zero rotations
    nonzero = angles > 1e-12
    nz_angles = angles[nonzero]
    nz_vecs = vecs[nonzero]

    # Normalize axis
    axes = nz_vecs / nz_angles[:, None]

    half = nz_angles / 2.0
    quats[nonzero, 3] = np.cos(half)
    quats[nonzero, :3] = axes * np.sin(half)[:, None]

    # Zero-rotation case → identity quaternion
    zero = ~nonzero
    quats[zero, 3] = 1.0  # (0,0,0,1)

    return quats


def axisangle_to_rot_6d(axisangle: np.ndarray) -> np.ndarray:
    """
    Converts axis-angle to 6d rotation representation
    axisangle: N, 3

    Returns:
        np.array: N, 6
    """
    quat = axisangle2quat(axisangle)
    return quat_to_rot_6d(quat)


def quat_to_rot_mat(quat: np.ndarray) -> np.ndarray:
    """
    Convert a quaternion to a rotation matrix
    quat: N, 4 (x, y, z, w)

    return: N, 3, 3
    """
    rot_mat = Rotation.from_quat(quat).as_matrix()
    return rot_mat


def quat_to_rot_6d(quat: np.ndarray) -> np.ndarray:
    """
    Convert a quaternion to 6d rotation representation
    quat: N, 4 (x, y, z, w)

    return: N, 6
    """
    rot_mat = Rotation.from_quat(quat).as_matrix()
    rot_6d = rot_mat[:, :2, :]  # N, 2, 3
    return rot_6d.reshape(-1, 6)  # N, 6


def rot_mat_to_rot_6d(rot_mat: np.ndarray) -> np.ndarray:
    """
    Convert a rotation matrix to 6d representation
    rot_mat: N, 3, 3

    return: N, 6
    """
    rot_6d = rot_mat[:, :2, :]  # N, 2, 3
    return rot_6d.reshape(-1, 6)  # N, 6


def rot_6d_to_rot_mat(rot_6d: np.ndarray) -> np.ndarray:
    """
    Convert a 6d representation to rotation matrix
    rot_6d: N, 6

    return: N, 3, 3
    """
    rot_6d = rot_6d.reshape(-1, 2, 3)
    # assert the first two vectors are orthogonal
    if not np.allclose(np.sum(rot_6d[:, 0] * rot_6d[:, 1], axis=-1), 0):
        rot_6d = gram_schmidt(rot_6d)

    rot_mat = np.zeros((rot_6d.shape[0], 3, 3))
    rot_mat[:, :2, :] = rot_6d
    rot_mat[:, 2, :] = np.cross(rot_6d[:, 0], rot_6d[:, 1])
    return rot_mat


def euler_to_rot_6d(euler: np.ndarray) -> np.ndarray:
    """
    Convert a 3D Euler angle to 6D rotation representation
    euler: N, 3

    return: N, 6
    """
    rot_mat = Rotation.from_euler("xyz", euler, degrees=False).as_matrix()
    return rot_mat_to_rot_6d(rot_mat)


def vec9d2T(vec: np.ndarray) -> np.ndarray:
    """
    Converts a (N, 9) vector of SE(3) poses (3D translation + 6D rotation) into (N, 4, 4) transformation matrices.

    Args:
        vec: (N, 9) array. Each row is [x, y, z, r1, r2, r3, r4, r5, r6],
             where r1–r6 is a 6D rotation representation.

    Returns:
        T: (N, 4, 4) SE(3) transformation matrices.
    """
    trans = vec[:, :3].reshape(-1, 3, 1)  # (N, 3, 1)
    rot6d = vec[:, 3:]  # (N, 6)
    rot_mat = rot_6d_to_rot_mat(rot6d)  # (N, 3, 3)

    N = vec.shape[0]
    T = np.tile(np.eye(4), (N, 1, 1))  # (N, 4, 4)
    T[:, :3, :3] = rot_mat
    T[:, :3, 3:4] = trans  # safer than [:,3] broadcasting

    return T


def T2vec9d(T):
    trans = T[:, :3, 3]
    rot = T[:, :3, :3]
    rot6d = rot_mat_to_rot_6d(rot)
    vec9d = np.concatenate([trans, rot6d], axis=-1)
    return vec9d


# A, B are (N, 4, 4) transformation matrices
def invert_transform(T):
    R = T[:, :3, :3]  # Rotation part
    t = T[:, :3, 3:]  # Translation part
    R_inv = np.transpose(R, axes=(0, 2, 1))
    t_inv = -R_inv @ t
    T_inv = np.zeros_like(T)
    T_inv[:, :3, :3] = R_inv
    T_inv[:, :3, 3] = t_inv[:, :, 0]
    T_inv[:, 3, 3] = 1.0
    return T_inv


def gram_schmidt(vectors: np.ndarray) -> np.ndarray:
    """
    Apply Gram-Schmidt process to a set of vectors
    vectors are indexed by rows

    vectors: batchsize, N, D

    return: batchsize, N, D
    """
    if len(vectors.shape) == 2:
        vectors = vectors[None]

    basis = np.zeros_like(vectors)
    basis[:, 0] = vectors[:, 0] / (
        np.linalg.norm(vectors[:, 0], axis=-1, keepdims=True) + 1e-8
    )
    for i in range(1, vectors.shape[1]):
        v = vectors[:, i]
        for j in range(i):
            v -= np.sum(v * basis[:, j], axis=-1, keepdims=True) * basis[:, j]
        basis[:, i] = v / (
            np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8
        )  # Normalize to avoid division by zero
    return basis


def combine_dicts(dlist, key):
    """
    Combine a list of dictionaries into a single dictionary.
    """
    d = {}
    for k in key:
        dk = [d[k] for d in dlist]
        d[k] = np.concatenate(dk, axis=0)
    return d


def load_json(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    return data


def convert_multi_step(data: torch.Tensor, num_pred_steps: int):
    """Chunk data for predicting data `num_pred_steps` steps into the future.
    The resulting data have shape (batch, data.shape[-2] - (num_pred_steps - 1), num_pred_steps, action_dim)
    For example: chunk_data([a_1, a_2, a_3, a_4, a_5], 3) ->
        [
            [a_1, a_2, a_3],
            [a_2, a_3, a_4],
            [a_3, a_4, a_5],
            [a_4, a_5, a_5],
            [a_5, a_5, a_5],
        ]
    adapted from https://github.com/octo-models/octo/blob/7480a2a90160122b7a02459fc6f56ceefa501ebf/octo/model/components/action_heads.py#L59
    """
    assert (
        data.ndim == 2
    ), f"Expected data to have shape (seq length, action_dim), but got shape {data.shape}"
    window_size = data.shape[0]
    chunk_window_size = window_size

    curr_step = torch.arange(chunk_window_size, device=data.device)
    action_offset = torch.arange(num_pred_steps, device=data.device)
    chunk_indices = torch.minimum(
        curr_step[:, None] + action_offset[None, :], torch.tensor(chunk_window_size - 1)
    )
    return data[chunk_indices]


def scale_action(
    action: torch.Tensor, stat: dict, type: Literal["minmax", "standard"] = "standard"
) -> torch.Tensor:
    """
    action: S, T, action_dim
    stat: dictionary
    """
    # move stats to action device
    for k, v in stat.items():
        stat[k] = v.to(action.device)
    action_dim = stat["min"].shape[0]
    if type == "minmax":
        action[..., :action_dim] = (action[..., :action_dim] - stat["min"]) / (
            stat["max"] - stat["min"]
        )
    elif type == "standard":
        action[..., :action_dim] = (action[..., :action_dim] - stat["mean"]) / stat[
            "std"
        ]
    return action


def unscale_action(
    action: torch.Tensor, stat: dict, type: Literal["minmax", "standard"] = "standard"
) -> torch.Tensor:
    """
    action: S, T, action_dim
    stat: dictionary
    """
    # move stats to action device
    for k, v in stat.items():
        stat[k] = v.to(action.device)
    action_dim = stat["min"].shape[0]
    if type == "minmax":
        action[..., :action_dim] = (
            action[..., :action_dim] * (stat["max"] - stat["min"]) + stat["min"]
        )
    elif type == "standard":
        action[..., :action_dim] = action[..., :action_dim] * stat["std"] + stat["mean"]
    return action


def _get_T_delta(state9d):
    """
    Compute relative SE(3) transform between consecutive frames in a sequence of 9D states.

    Args:
        state9d: (N, 9) array. Each row contains a 3D translation and 6D rotation representation.

    Returns:
        vec9d: (N, 9) array. The delta in translation and rotation (as 6D) between frames t and t+1.
               The last frame has identity delta (i.e., no motion).
    """
    T = vec9d2T(state9d)  # (N, 4, 4)
    T_cur = T[:-1]  # (N-1, 4, 4)
    T_fur = T[1:]  # (N-1, 4, 4)
    delta_T = invert_transform(T_cur) @ T_fur  # (N-1, 4, 4)

    # Pad with identity for the last frame to keep output length == input length
    identity = np.eye(4)[None, :, :]  # (1, 4, 4)
    delta_T = np.concatenate([delta_T, identity], axis=0)  # (N, 4, 4)

    rot6d = rot_mat_to_rot_6d(delta_T[:, :3, :3])  # (N, 6)
    trans = delta_T[:, :3, 3]  # (N, 3)

    vec9d = np.concatenate([trans, rot6d], axis=-1)  # (N, 9)
    return vec9d


def random_wrist_mask(rgbs, camera_names, prob, gen):
    """Randomly zero 1-2 wrist views in a list of per-view image tensors.

    A view counts as a 'wrist' if its camera name contains 'wrist' or 'hand'.
    Ego / non-wrist views are always kept. With probability `prob`, mask 1
    wrist (or, when >=2 wrists exist, 2 wrists with 50% chance). Used to make
    the latent-action inverse model robust to missing wrist cameras.
    Returns `rgbs` (modified in place).
    """
    import torch as _torch

    if prob is None or prob <= 0.0:
        return rgbs
    wrist_idx = [
        i for i, n in enumerate(camera_names)
        if ("wrist" in str(n).lower() or "hand" in str(n).lower())
    ]
    if not wrist_idx:
        return rgbs
    if _torch.rand(1, generator=gen).item() >= prob:
        return rgbs
    if len(wrist_idx) >= 2 and _torch.rand(1, generator=gen).item() < 0.5:
        k = 2
    else:
        k = 1
    chosen = _torch.randperm(len(wrist_idx), generator=gen)[:k].tolist()
    for j in chosen:
        i = wrist_idx[j]
        rgbs[i] = _torch.zeros_like(rgbs[i])
    return rgbs


def pad_to_num_views(rgbs, names, num_views):
    """Pad a per-view tensor list up to num_views with zero views (name 'pad').

    Ensures heterogeneous datasets stack to an identical grid so multi-dataset
    batches collate and share one rgb_pos_embed. Pads are NOT counted as wrists
    by random_wrist_mask (name has no 'wrist'/'hand'). Truncates if too many.
    """
    import torch as _torch

    if num_views is None:
        return rgbs, names
    rgbs, names = list(rgbs), list(names)
    rgbs = rgbs[:num_views]
    names = names[:num_views]
    while len(rgbs) < num_views:
        rgbs.append(_torch.full_like(rgbs[0], -1.0))  # black pad (=-1 in [-1,1]); gray(0) read as washed
        names.append(f"pad{len(names)}")
    return rgbs, names
