import hashlib
import json
import logging
import math
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Union

import h5py
import numpy as np
import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset
from tqdm import tqdm

import robot_wm.utils.distributed as dist

logger = logging.getLogger(__name__)
NORMALIZABLE_KEYS = ["state", "actions"]


def _h5py_to_dict(
    file: Union[h5py.File, h5py.Group], max_load_dim: Union[int, None] = None
) -> dict:
    def _str_or_numpy(
        item: h5py.Dataset, encoding: str = "utf-8"
    ) -> Union[str, np.ndarray]:
        return item.asstr(encoding)[()] if item.dtype == "object" else np.array(item)

    output = {}
    for key in file:
        if isinstance(file[key], h5py.Dataset):
            if max_load_dim is not None and len(file[key].shape) > max_load_dim:
                output[key] = file[key]
            else:
                output[key] = _str_or_numpy(file[key])
        else:
            output[key] = _h5py_to_dict(file[key])
    return output


# Utility function to compute statistics on a numpy array
def compute_stats(values: np.ndarray) -> dict:
    stats = {
        "raw_min": np.min(values, axis=0).tolist(),
        "raw_max": np.max(values, axis=0).tolist(),
        "raw_mean": np.mean(values, axis=0).tolist(),
        "raw_std": np.std(values, axis=0).tolist(),
        "raw_q1": np.quantile(values, 0.01, axis=0).tolist(),
        "raw_q99": np.quantile(values, 0.99, axis=0).tolist(),
    }
    q1, q99 = np.array(stats["raw_q1"]), np.array(stats["raw_q99"])
    mask = (values >= q1) & (values <= q99)
    values = np.where(mask, values, 0)
    # extend stats with stats of the masked values
    stats["min"] = np.min(values, axis=0).tolist()
    stats["max"] = np.max(values, axis=0).tolist()
    stats["mean"] = np.mean(values, axis=0).tolist()
    stats["std"] = np.std(values, axis=0).tolist()
    stats["q1"] = np.quantile(values, 0.01, axis=0).tolist()
    stats["q99"] = np.quantile(values, 0.99, axis=0).tolist()
    return stats


def compute_stats_path(
    manifest_path: Union[str, Path, list],
    keys=None,
    stats_save_dir: Optional[Path] = None,
    interpolation_step: Optional[int] = None,
) -> Path:
    if not isinstance(manifest_path, (str, Path)):
        manifest_path = "".join(manifest_path)
        assert (
            stats_save_dir is not None
        ), "stats_save_dir must be provided when manifest_path is a list"
        hash_input = str(manifest_path)
        save_dir = Path(stats_save_dir)
    else:
        manifest_path = Path(manifest_path)
        hash_input = str(manifest_path.resolve())
        save_dir = (
            Path(stats_save_dir) if stats_save_dir is not None else manifest_path.parent
        )

    if keys is not None:
        for key in keys:
            hash_input += f"_{key}"

    if interpolation_step is not None:
        hash_input += f"_interp_{interpolation_step}"
    hash_digest = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()[:10]

    return save_dir / f"{hash_digest}.json"


class Dataset(ABC, IterableDataset, Stateful):
    def __init__(
        self,
        seed: int = 0,
        infinite: bool = False,
        transform: Optional[Stateful] = None,
    ):
        self._seed = seed
        self._infinite = infinite
        self._transform = transform

        self._start_idx = 0
        self._gen = torch.Generator()

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def _get_length(self) -> int:
        pass

    @abstractmethod
    def _get_sample(self, index: int) -> dict[str, Any]:
        pass

    def __iter__(self):
        self._gen.manual_seed(self._seed)

        # shard dataset across processes
        process_id, num_processes = 0, 1
        if dist.is_initialized():
            process_id = dist.get_global_rank()
            num_processes = dist.get_world_size()

        N = self._get_length()
        total = int(math.ceil(N / num_processes)) * num_processes
        process_indices = torch.randperm(total, generator=self._gen) % N
        process_indices = process_indices[process_id:total:num_processes]

        # shard dataset across workers
        worker_id, num_workers = 0, 1
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        N = len(process_indices)
        total = int(math.ceil(N / num_workers)) * num_workers
        worker_indices = torch.randperm(total, generator=self._gen) % N
        worker_indices = worker_indices[worker_id:total:num_workers]

        # get the subset of `process_indices` for this worker
        final_indices = process_indices[worker_indices]

        # generate worker data
        while True:
            for idx in range(self._start_idx, len(final_indices)):
                self._start_idx += 1
                index = final_indices[idx]
                yield self._get_sample(index)
            if not self._infinite:
                break
            self._start_idx = 0
            logger.log(level=logging.DEBUG, msg=f"Finished {self.name}; restarting")

    def state_dict(self) -> dict[str, Any]:
        output = {
            "_start_idx": self._start_idx,
            "_gen": self._gen.get_state(),
        }
        if self._transform is not None:
            output["_transform"] = self._transform.state_dict()
        return output

    def load_state_dict(self, state_dict: dict[str, Any]):
        self._start_idx = state_dict["_start_idx"]
        self._gen.set_state(state_dict["_gen"])
        if self._transform is not None:
            self._transform.load_state_dict(state_dict["_transform"])

    def _load_manifest(self, manifest_path: Union[str, Path]) -> list[Path]:
        pass

    def compute_and_save_stats(
        self,
        statistic_manifest,
        keys: list[str],
        stats_save_dir: Optional[Path] = None,
        overwrite_camera_motion=False,
        interpolation_step: Optional[int] = None,
    ) -> Path:
        stats_path = compute_stats_path(
            statistic_manifest, keys, stats_save_dir, interpolation_step
        )

        if os.path.exists(stats_path):
            print(f"[INFO] Stats file already exists: {stats_path}")
            return stats_path  # You could load it here if needed

        print("[INFO] Computing stats... and saving to", stats_path)
        self._paths = self._load_manifest(statistic_manifest)

        # Validate keys
        for key in keys:
            assert key in NORMALIZABLE_KEYS

        # Initialize data collection for all keys
        all_values = {key: [] for key in keys}

        # Single pass through dataset to collect all required keys
        dataset_length = self.__len__()
        for idx in tqdm(range(dataset_length), desc="Processing samples"):
            sample = self._get_sample(idx)
            if isinstance(sample, tuple):
                sample = sample[0]

            # Extract all required keys in one pass
            for key in keys:
                arr = sample[key].view(-1, sample[key].shape[-1]).numpy()
                all_values[key].append(arr)

        # Compute stats for each key
        all_stats = {}
        for key in keys:
            print(f"[INFO] Computing statistics for key: {key}")
            concatenated_values = np.concatenate(all_values[key], axis=0)
            all_stats[key] = compute_stats(concatenated_values)

            if overwrite_camera_motion:
                for k in all_stats[key].keys():
                    all_stats[key][k][-9:] = all_stats[key][k][
                        :9
                    ]  # overwrite last 9 values with the first 9 values

        with open(stats_path, "w") as f:
            json.dump(all_stats, f, indent=2)

        print(f"[INFO] Stats saved to {stats_path}")
        return stats_path


class Transform(ABC, Stateful):
    @abstractmethod
    def __call__(self, episode: dict[str, Any]) -> dict[str, Any]:
        pass
