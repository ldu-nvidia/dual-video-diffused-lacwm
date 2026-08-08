#!/usr/bin/env python3
"""Run the sealed val64 CSIP aligned/control evaluation exactly once.

The fixed update-400 checkpoint is sealed before this tool may open validation
latents or actions.  Predictions from each clean real-video spectral feature
are scored against its aligned action target, a donor from its deterministic
disjoint episode pair, the raw-no-action target, and the sign-inverted target.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robot_wm.modeling.dual_diffusion.causal_spectral_probe import (  # noqa: E402
    ActionPCATransform,
    FrozenCausalSpectralProbe,
    angle_neutral_spectral_features,
    causal_spectral_features,
    control_targets,
    episode_disjoint_pair_donors,
)
from tools import csip_contract as contract  # noqa: E402
from tools.csip_workflow import validate_seal  # noqa: E402


FEATURE_BATCH_SIZE = 16


def _features(full: Any, history: Any, device: Any) -> Any:
    import torch

    values = []
    with torch.inference_mode():
        for start in range(0, len(full), FEATURE_BATCH_SIZE):
            stop = min(start + FEATURE_BATCH_SIZE, len(full))
            values.append(
                causal_spectral_features(
                    torch.from_numpy(full[start:stop]).to(
                        device=device, dtype=torch.float32
                    ),
                    torch.from_numpy(history[start:stop]).to(
                        device=device, dtype=torch.float32
                    ),
                ).cpu()
            )
    return torch.cat(values, dim=0)


def _metric_rows(prediction: Any, targets: dict[str, Any]) -> dict[str, dict[str, Any]]:
    import torch

    result: dict[str, dict[str, Any]] = {}
    for name, target in targets.items():
        mse = (prediction - target).square().mean(dim=1)
        cosine = torch.nn.functional.cosine_similarity(
            prediction, target, dim=1, eps=1e-8
        )
        result[name] = {
            "mse": [float(value) for value in mse.tolist()],
            "cosine": [float(value) for value in cosine.tolist()],
            "mean_mse": float(mse.mean().item()),
            "mean_cosine": float(cosine.mean().item()),
        }
    return result


def command_evaluate(args: argparse.Namespace) -> int:
    import numpy as np

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if os.environ["CUBLAS_WORKSPACE_CONFIG"] not in {":4096:8", ":16:8"}:
        raise contract.CSIPContractError(
            "deterministic CSIP evaluation requires a supported CUBLAS_WORKSPACE_CONFIG"
        )
    import torch

    seal, sealed_registration = validate_seal(args.seal)
    registration_path = seal["registration"]["path"]
    registration = contract.validate_registration(
        registration_path,
        require_train_cache=True,
        require_validation_cache=True,
        open_validation=True,
    )
    if registration["identity_sha256"] != sealed_registration["identity_sha256"]:
        raise contract.CSIPContractError("sealed registration changed")
    contract.current_python_matches(registration["runtime"]["python"])
    output = Path(registration["planned_paths"]["evaluation"])
    if output.exists() or output.is_symlink():
        raise contract.CSIPContractError("sealed evaluation output must be fresh")

    validation_metadata_path = Path(
        registration["planned_paths"]["validation_latent_metadata"]
    )
    metadata = contract.validate_latent_cache(
        validation_metadata_path, registration=registration, split="validation"
    )
    cache_seal = metadata.get("checkpoint_seal")
    observed_seal_record = {
        **contract.file_record(args.seal, "checkpoint seal"),
        "identity_sha256": seal["identity_sha256"],
    }
    if cache_seal != observed_seal_record:
        raise contract.CSIPContractError(
            "validation latents were not extracted under this fixed checkpoint seal"
        )
    full, history = contract.load_latent_cache_arrays(metadata)
    actions_path = registration["datasets"]["validation"]["arrays"]["actions"]["path"]
    actions_np = np.load(actions_path, mmap_mode="r", allow_pickle=False)
    if tuple(actions_np.shape) != (64, 13, 5, 23) or str(actions_np.dtype) != "float32":
        raise contract.CSIPContractError("validation action cache geometry differs")
    manifest_path = registration["datasets"]["validation"]["manifest"]["path"]
    rows = contract.manifest_rows(manifest_path, split="val", count=64)
    episode_ids = [str(row["episode_dir"]) for row in rows]
    donor_indexes, donor_pairs = episode_disjoint_pair_donors(episode_ids)
    if any(
        episode_ids[index] == episode_ids[int(donor_indexes[index])]
        for index in range(contract.EXPECTED_VALIDATION_CLIPS)
    ):
        raise contract.CSIPContractError("validation donor is not episode-disjoint")
    if len(donor_pairs) != contract.EXPECTED_VALIDATION_PAIR_BLOCKS or any(
        int(donor_indexes[left]) != right or int(donor_indexes[right]) != left
        for left, right in donor_pairs
    ):
        raise contract.CSIPContractError(
            "validation donor pairs are not disjoint swaps"
        )
    pair_blocks = {
        member: block
        for block, pair in enumerate(donor_pairs)
        for member in pair
    }
    if set(pair_blocks) != set(range(contract.EXPECTED_VALIDATION_CLIPS)):
        raise contract.CSIPContractError("validation donor pairs do not cover val64")

    if not torch.cuda.is_available():
        raise contract.CSIPContractError("CSIP sealed evaluation requires CUDA")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.use_deterministic_algorithms(True)
    full_features = _features(full, history, device)
    features = {
        "full": full_features,
        "angle_neutral": angle_neutral_spectral_features(full_features),
    }
    checkpoint_path = contract.verify_file_record(
        seal["checkpoint"], "sealed checkpoint"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    train_metadata = contract.validate_latent_cache(
        registration["planned_paths"]["train_latent_metadata"],
        registration=registration,
        split="train",
    )
    contract.validate_checkpoint_payload(
        checkpoint,
        registration=registration,
        train_latent_cache_identity_sha256=train_metadata["identity_sha256"],
    )
    models = {}
    for name in ("full", "angle_neutral"):
        model = FrozenCausalSpectralProbe(
            hidden_dim=int(checkpoint["model_hidden_dim"])
        )
        model.load_state_dict(checkpoint["model_states"][name], strict=True)
        models[name] = model.to(device).eval()
    pca = ActionPCATransform.from_state_dict(checkpoint["action_pca"])
    actions = torch.from_numpy(np.array(actions_np, copy=True)).float()
    targets = control_targets(actions, pca, donor_indexes)
    with torch.inference_mode():
        predictions = {
            name: model(features[name].to(device)).cpu()
            for name, model in models.items()
        }
    metrics = {
        name: _metric_rows(prediction, targets)
        for name, prediction in predictions.items()
    }

    clip_records = []
    for index, row in enumerate(rows):
        donor = int(donor_indexes[index])
        clip_records.append(
            {
                "auxiliary_index": index,
                "clip_id": row["clip_id"],
                "episode_dir": row["episode_dir"],
                "donor_index": donor,
                "donor_pair_block": pair_blocks[index],
                "donor_clip_id": rows[donor]["clip_id"],
                "donor_episode_dir": rows[donor]["episode_dir"],
                "metrics": {
                    probe: {
                        condition: {
                            "mse": metrics[probe][condition]["mse"][index],
                            "cosine": metrics[probe][condition]["cosine"][index],
                        }
                        for condition in metrics[probe]
                    }
                    for probe in metrics
                },
            }
        )
    evaluation = contract.with_identity(
        {
            "schema_version": contract.SCHEMA_VERSION,
            "kind": contract.EVALUATION_KIND,
            "created_at_utc": contract.now_utc(),
            "status": "sealed_val64_complete",
            "registration": contract.registration_file_record(registration_path),
            "registration_identity_sha256": registration["identity_sha256"],
            "checkpoint_seal": observed_seal_record,
            "checkpoint": seal["checkpoint"],
            "validation_latent_metadata": contract.file_record(
                validation_metadata_path, "validation latent metadata"
            ),
            "validation_latent_cache_identity_sha256": metadata["identity_sha256"],
            "validation_clips": contract.EXPECTED_VALIDATION_CLIPS,
            "donor_control": {
                "kind": "episode_disjoint_two_episode_pairs",
                "pairing_rule": "adjacent_immutable_manifest_indexes_swap_targets",
                "pair_count": contract.EXPECTED_VALIDATION_PAIR_BLOCKS,
                "pairs": [
                    {
                        "pair_block": block,
                        "left_index": left,
                        "right_index": right,
                    }
                    for block, (left, right) in enumerate(donor_pairs)
                ],
                "self_donors": 0,
                "same_episode_donors": 0,
                "overlapping_pair_blocks": 0,
            },
            "probes": list(metrics),
            "conditions": list(metrics["full"]),
            "summary": {
                probe: {
                    condition: {
                        "mean_mse": values["mean_mse"],
                        "mean_cosine": values["mean_cosine"],
                    }
                    for condition, values in probe_values.items()
                }
                for probe, probe_values in metrics.items()
            },
            "angle_comparator": {
                "kind": "matched_9216_input_phasors_replaced_by_support_zero",
                "support_encoding": "each_masked_unit_phasor_becomes_mask_zero",
                "same_initialization": True,
                "same_batches_optimizer_updates_targets_architecture": True,
            },
            "clips": clip_records,
            "validation_used_for_training_or_checkpoint_selection": False,
            "protected_test_paths_accepted": False,
            "protected_test_clips_read": 0,
        }
    )
    contract.exclusive_json(output, evaluation)
    print(
        json.dumps(
            {
                "evaluation": str(output),
                "identity_sha256": evaluation["identity_sha256"],
                "summary": evaluation["summary"],
            },
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal", type=Path, required=True)
    return parser


def main() -> int:
    return command_evaluate(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
