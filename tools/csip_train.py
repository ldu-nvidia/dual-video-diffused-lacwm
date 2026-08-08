#!/usr/bin/env python3
"""Fit the fixed Phase-0 CSIP probe on fit448 and monitor cal64 only.

The tool has no validation argument and refuses to start if the registered
validation latent cache already exists.  PCA and all trainable weights use
only fit448.  Calibration64 is evaluated at fixed updates but never selects a
checkpoint, threshold, or stopping time.  The only checkpoint is update 400.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robot_wm.modeling.dual_diffusion.causal_spectral_probe import (  # noqa: E402
    FrozenCausalSpectralProbe,
    action_descriptor,
    causal_spectral_features,
    fit_action_pca,
    phase0_partition_indexes,
)
from tools import csip_contract as contract  # noqa: E402


CALIBRATION_UPDATES = (0, 100, 200, 300, 400)
FEATURE_BATCH_SIZE = 16


def _atomic_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists() or path.is_symlink():
        raise contract.CSIPContractError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise contract.CSIPContractError("checkpoint temporary path exists")
    with temporary.open("xb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _metrics(prediction: "Any", target: "Any") -> dict[str, float]:
    import torch

    mse_per_clip = (prediction - target).square().mean(dim=1)
    cosine_per_clip = torch.nn.functional.cosine_similarity(
        prediction, target, dim=1, eps=1e-8
    )
    return {
        "mse": float(mse_per_clip.mean().item()),
        "cosine": float(cosine_per_clip.mean().item()),
    }


def _extract_features(full: "Any", history: "Any", device: "Any") -> "Any":
    import torch

    batches = []
    with torch.inference_mode():
        for start in range(0, len(full), FEATURE_BATCH_SIZE):
            stop = min(start + FEATURE_BATCH_SIZE, len(full))
            full_batch = torch.from_numpy(full[start:stop]).to(
                device=device, dtype=torch.float32
            )
            history_batch = torch.from_numpy(history[start:stop]).to(
                device=device, dtype=torch.float32
            )
            batches.append(causal_spectral_features(full_batch, history_batch).cpu())
    return torch.cat(batches, dim=0)


def _initialize_wandb(registration: dict[str, Any]) -> tuple[Any, Any]:
    try:
        import wandb
    except ImportError as exc:
        raise contract.CSIPContractError(
            "W&B is required for any CSIP training run"
        ) from exc
    record = registration["wandb"]
    if (
        record.get("entity") != contract.EXPECTED_ENTITY
        or record.get("project") != contract.EXPECTED_PROJECT
        or str(record.get("access", "")).upper() != "PRIVATE"
        or record.get("group") is not None
        or record.get("mode") != "online"
        or record.get("private_project_acknowledged") is not True
        or record.get("viewer_username") != contract.EXPECTED_ENTITY
        or record.get("user_requested_email") != "ldu@nvidia.edu"
        or record.get("authenticated_email_matches_user_request")
        != (record.get("viewer_email") == "ldu@nvidia.edu")
    ):
        raise contract.CSIPContractError("private personal W&B contract differs")
    # Privacy and authenticated personal ownership are execution-time
    # properties, not merely registration-time facts.
    from tools.csip_workflow import _wandb_attestation

    live = _wandb_attestation(contract.EXPECTED_ENTITY, contract.EXPECTED_PROJECT)
    if live.get("viewer_username") != record.get("viewer_username") or live.get(
        "viewer_email"
    ) != record.get("viewer_email"):
        raise contract.CSIPContractError(
            "authenticated W&B identity changed after registration"
        )
    wandb_dir = Path(registration["planned_paths"]["wandb_local_dir"])
    wandb_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    run = wandb.init(
        entity=contract.EXPECTED_ENTITY,
        project=contract.EXPECTED_PROJECT,
        name=record["run_name"],
        id=record["run_id"],
        group=None,
        resume="never",
        mode="online",
        dir=str(wandb_dir),
        config={
            "registration_identity_sha256": registration["identity_sha256"],
            "fit_clips": 448,
            "calibration_clips": 64,
            "fixed_updates": contract.EXPECTED_UPDATES,
            "batch_size": contract.EXPECTED_BATCH_SIZE,
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
            "seed": contract.EXPECTED_SEED,
            "validation_clips_read": 0,
            "protected_test_clips_read": 0,
        },
        settings=wandb.Settings(init_timeout=120),
    )
    if run is None:
        raise contract.CSIPContractError("W&B did not create a run")
    if run.id != record["run_id"]:
        raise contract.CSIPContractError("W&B run ID differs from registration")
    return wandb, run


def command_train(args: argparse.Namespace) -> int:
    import numpy as np

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if os.environ["CUBLAS_WORKSPACE_CONFIG"] not in {":4096:8", ":16:8"}:
        raise contract.CSIPContractError(
            "deterministic CSIP fitting requires a supported CUBLAS_WORKSPACE_CONFIG"
        )
    import torch

    registration = contract.validate_registration(
        args.registration,
        require_train_cache=True,
        open_validation=False,
    )
    contract.current_python_matches(registration["runtime"]["python"])
    validation_metadata = Path(
        registration["planned_paths"]["validation_latent_metadata"]
    )
    if validation_metadata.exists() or validation_metadata.is_symlink():
        raise contract.CSIPContractError(
            "validation latent cache must remain sealed until checkpoint creation"
        )
    checkpoint_path = Path(registration["planned_paths"]["checkpoint"])
    report_path = Path(registration["planned_paths"]["training_report"])
    if checkpoint_path.exists() or report_path.exists():
        raise contract.CSIPContractError("training outputs must be fresh")
    cache_path = Path(registration["planned_paths"]["train_latent_metadata"])
    latent_metadata = contract.validate_latent_cache(
        cache_path, registration=registration, split="train"
    )
    full, history = contract.load_latent_cache_arrays(latent_metadata)
    actions_path = registration["datasets"]["train"]["arrays"]["actions"]["path"]
    actions_np = np.load(actions_path, mmap_mode="r", allow_pickle=False)
    if (
        tuple(actions_np.shape) != (512, 13, 5, 23)
        or str(actions_np.dtype) != "float32"
    ):
        raise contract.CSIPContractError("training action cache geometry differs")

    if not torch.cuda.is_available():
        raise contract.CSIPContractError("CSIP fitting requires CUDA")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    random.seed(contract.EXPECTED_SEED)
    np.random.seed(contract.EXPECTED_SEED)
    torch.manual_seed(contract.EXPECTED_SEED)
    torch.cuda.manual_seed_all(contract.EXPECTED_SEED)
    torch.use_deterministic_algorithms(True)

    features = _extract_features(full, history, device)
    if tuple(features.shape) != (512, 9216):
        raise contract.CSIPContractError("training spectral feature geometry differs")
    actions = torch.from_numpy(np.array(actions_np, copy=True)).float()
    fit_indexes = torch.tensor(phase0_partition_indexes(512, "fit"), dtype=torch.long)
    calibration_indexes = torch.tensor(
        phase0_partition_indexes(512, "calibration"), dtype=torch.long
    )
    pca = fit_action_pca(action_descriptor(actions[fit_indexes]))
    targets = pca.transform_actions(actions)
    fit_features = features[fit_indexes].to(device)
    fit_targets = targets[fit_indexes].to(device)
    calibration_features = features[calibration_indexes].to(device)
    calibration_targets = targets[calibration_indexes].to(device)

    model = FrozenCausalSpectralProbe().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(contract.EXPECTED_SEED)
    calibration: dict[str, Any] = {}
    wandb, run = _initialize_wandb(registration)
    try:
        model.eval()
        with torch.inference_mode():
            calibration["0"] = _metrics(
                model(calibration_features), calibration_targets
            )
        wandb.log(
            {
                "calibration/mse": calibration["0"]["mse"],
                "calibration/cosine": calibration["0"]["cosine"],
            },
            step=0,
        )
        permutation = torch.empty(0, dtype=torch.long)
        cursor = 448
        for update in range(1, contract.EXPECTED_UPDATES + 1):
            if cursor + contract.EXPECTED_BATCH_SIZE > 448:
                permutation = torch.randperm(448, generator=generator)
                cursor = 0
            batch_indexes = permutation[cursor : cursor + contract.EXPECTED_BATCH_SIZE]
            cursor += contract.EXPECTED_BATCH_SIZE
            model.train()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(fit_features[batch_indexes.to(device)])
            target = fit_targets[batch_indexes.to(device)]
            loss = torch.nn.functional.mse_loss(prediction, target)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("CSIP fit loss became non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            wandb.log({"fit/mse": float(loss.item())}, step=update)
            if update in CALIBRATION_UPDATES:
                model.eval()
                with torch.inference_mode():
                    metrics = _metrics(model(calibration_features), calibration_targets)
                calibration[str(update)] = metrics
                wandb.log(
                    {
                        "calibration/mse": metrics["mse"],
                        "calibration/cosine": metrics["cosine"],
                    },
                    step=update,
                )
        model.eval()
        with torch.inference_mode():
            fit_metrics = _metrics(model(fit_features), fit_targets)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        checkpoint = {
            "schema_version": contract.SCHEMA_VERSION,
            "kind": contract.CHECKPOINT_KIND,
            "registration_identity_sha256": registration["identity_sha256"],
            "train_latent_cache_identity_sha256": latent_metadata["identity_sha256"],
            "completed_updates": contract.EXPECTED_UPDATES,
            "selected_update": contract.EXPECTED_UPDATES,
            "selection_rule": "fixed_final_update_not_metric_selected",
            "seed": contract.EXPECTED_SEED,
            "fit_indexes": fit_indexes,
            "calibration_indexes": calibration_indexes,
            "action_pca": pca.state_dict(),
            "model_state": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "optimizer_state": optimizer.state_dict(),
            "model_hidden_dim": 256,
            "spectral_feature_dim": 9216,
            "action_target_dim": 16,
            "fit_metrics": fit_metrics,
            "calibration_metrics": calibration,
            "wandb_run_id": run.id,
            "validation_clips_read": 0,
            "protected_test_clips_read": 0,
        }
        _atomic_checkpoint(checkpoint_path, checkpoint)
        checkpoint_sha256 = contract.sha256_file(checkpoint_path)
        report = contract.with_identity(
            {
                "schema_version": contract.SCHEMA_VERSION,
                "kind": "csip_phase0_training_report",
                "created_at_utc": contract.now_utc(),
                "status": "fixed_update_400_complete",
                "registration": contract.registration_file_record(args.registration),
                "registration_identity_sha256": registration["identity_sha256"],
                "train_latent_metadata": contract.file_record(
                    cache_path, "train latent metadata"
                ),
                "train_latent_cache_identity_sha256": latent_metadata[
                    "identity_sha256"
                ],
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha256,
                "completed_updates": contract.EXPECTED_UPDATES,
                "selection_rule": "fixed_final_update_not_metric_selected",
                "fit_clips": 448,
                "calibration_clips": 64,
                "fit_metrics": fit_metrics,
                "calibration_metrics": calibration,
                "parameter_count": parameter_count,
                "wandb": {
                    "entity": contract.EXPECTED_ENTITY,
                    "project": contract.EXPECTED_PROJECT,
                    "group": None,
                    "run_id": run.id,
                    "url": run.url,
                },
                "validation_clips_read": 0,
                "protected_test_clips_read": 0,
            }
        )
        contract.exclusive_json(report_path, report)
        run.summary["checkpoint_sha256"] = checkpoint_sha256
        run.summary["fixed_update"] = contract.EXPECTED_UPDATES
        run.summary["fit_mse"] = fit_metrics["mse"]
        run.summary["calibration_mse"] = calibration["400"]["mse"]
        print(
            json.dumps(
                {
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_sha256": checkpoint_sha256,
                    "training_report": str(report_path),
                    "fixed_updates": contract.EXPECTED_UPDATES,
                    "validation_clips_read": 0,
                },
                sort_keys=True,
            )
        )
    finally:
        wandb.finish()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    return parser


def main() -> int:
    return command_train(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
