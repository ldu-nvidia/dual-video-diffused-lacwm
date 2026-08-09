#!/usr/bin/env python3
"""Isolated exact-6560866 native-parent reference sampler.

This script intentionally does not import the current Stage-1 package.  A
coordinator starts it in a separate Python process, points all model imports at
an independently registered clean worktree for commit 6560866, and gives it a
small target-blind tensor bundle containing only observed train RGB history,
candidate actions, morphology, and immutable sample IDs.  The resulting
latents/decoded tensors are compared bit-for-bit with the current registered
evaluation adapter before validation data can open.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


PARENT_COMMIT = "656086686dae723c942a4209a9d71cdb17ed6ccc"
PARENT_SNAPSHOT_SHA256 = (
    "de65e832c56f82be1472edb1fd789e16d3a6c8a7adc9b1f31306779951cb463a"
)
PARENT_RUN_IDENTITY = (
    "d79c3699f0c68dcd17321fcb0ea2846fca4d96f8c815f91dff02a19d3140787f"
)
PARENT_CONFIG_SHA256 = (
    "ae3ffd27146883917472b828c18568b72cfc7c6f2888fbca3eaa2e980a8ffd38"
)
NFE_GRID = (1, 2, 4)


class HistoricalReferenceError(RuntimeError):
    """The isolated historical source, input, model, or sampler changed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        raise HistoricalReferenceError(
            f"historical git {' '.join(args)} failed: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return completed.stdout.strip()


def _install_historical_imports(
    repo: Path, *, videox: Path, wan: Path
) -> None:
    project = repo / "projects" / "latent_action_models"
    shim = repo / "tools" / "env" / "videox_shim"
    for root in reversed((str(repo), str(project), str(shim), str(videox))):
        while root in sys.path:
            sys.path.remove(root)
        sys.path.insert(0, root)
    os.environ["WAN_DIR"] = str(wan)
    os.environ["VIDEOX_HOME"] = str(videox)


def _to_uint8(value: Any) -> Any:
    import torch

    return (
        ((value.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 127.5)
        .round()
        .to(torch.uint8)
    )


def command_reference(args: argparse.Namespace) -> int:
    repo = args.implementation_repo.expanduser().resolve(strict=True)
    config_path = args.parent_config.expanduser().resolve(strict=True)
    snapshot_path = args.parent_snapshot.expanduser().resolve(strict=True)
    input_path = args.input.expanduser().resolve(strict=True)
    output = args.output.expanduser().absolute()
    wan = args.wan_dir.expanduser().resolve(strict=True)
    videox = args.videox_home.expanduser().resolve(strict=True)
    if (
        _git(repo, "rev-parse", "--show-toplevel") != str(repo)
        or _git(repo, "rev-parse", "HEAD") != PARENT_COMMIT
        or _git(repo, "status", "--porcelain", "--untracked-files=all")
        or output.exists()
        or output.is_symlink()
        or not output.parent.is_dir()
        or _sha256(config_path) != PARENT_CONFIG_SHA256
        or _sha256(snapshot_path) != PARENT_SNAPSHOT_SHA256
    ):
        raise HistoricalReferenceError(
            "historical source/config/snapshot/output contract differs"
        )
    partial = output.with_name(output.name + ".partial")
    if partial.exists() or partial.is_symlink():
        raise HistoricalReferenceError("historical partial output already exists")

    _install_historical_imports(repo, videox=videox, wan=wan)
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    if not torch.cuda.is_available():
        raise HistoricalReferenceError("historical reference requires B200")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    if "B200" not in torch.cuda.get_device_properties(device).name.upper():
        raise HistoricalReferenceError("historical reference requires B200")
    config = OmegaConf.load(config_path)
    model_config = config.trainer.model
    dual = model_config.dual_diffusion
    if (
        str(config.name)
        != "vjepa2-faithful-cascade-20260730-seed1234-6560866-v1-VPM"
        or int(config.seed) != 1234
        or "preserve_zero_support" in dual
        or "preserve_zero_support" in model_config.forward_model.dual_diffusion
        or bool(dual.condition_on_tf)
        or str(dual.condition_mode) != "off"
        or str(dual.schedule_mode) != "aligned"
        or int(dual.tf_channels) != 64
        or float(dual.tf_loss_weight) != 0.0
        or not bool(dual.parameter_matched_control)
        or int(dual.evaluation_noise_seed) != 20260729
    ):
        raise HistoricalReferenceError("historical resolved model config differs")
    model = instantiate(model_config)
    snapshot = torch.load(
        snapshot_path, map_location="cpu", mmap=True, weights_only=True
    )
    state = snapshot.get("model")
    if (
        snapshot.get("snapshot_schema_version") != 3
        or snapshot.get("world_size") != 8
        or snapshot.get("gradient_accumulation_steps") != 1
        or snapshot.get("_start_iter") != 1000
        or snapshot.get("run_identity_sha256") != PARENT_RUN_IDENTITY
        or not isinstance(state, Mapping)
        or len(state) != 1686
        or any(key in snapshot for key in ("ema", "model_ema", "ema_model"))
    ):
        raise HistoricalReferenceError("historical parent snapshot differs")
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise HistoricalReferenceError("historical strict state load failed")
    del state, snapshot
    model = model.to(device=device).eval()
    if hasattr(model.forward_model.tf_token_adapter, "preserve_zero_support"):
        raise HistoricalReferenceError(
            "historical adapter unexpectedly contains the later runtime option"
        )

    bundle = torch.load(input_path, map_location="cpu", weights_only=True)
    if (
        not isinstance(bundle, Mapping)
        or set(bundle)
        != {"history_rgb", "actions", "morphology_index", "sample_ids"}
    ):
        raise HistoricalReferenceError("target-blind input bundle differs")
    history = bundle["history_rgb"].to(device=device)
    actions = bundle["actions"].to(device=device)
    morphology = bundle["morphology_index"].to(device=device)
    sample_ids = bundle["sample_ids"].to(device=device)
    if (
        tuple(history.shape) != (1, 5, 3, 180, 960)
        or tuple(actions.shape) != (1, 13, 5, 157)
        or tuple(morphology.shape) != (1,)
        or tuple(sample_ids.shape) != (1,)
    ):
        raise HistoricalReferenceError("target-blind input tensor geometry differs")

    results = {}
    total_wan_calls = 0
    for nfe in NFE_GRID:
        model.evaluation_condition_sources = ("off",)
        model.evaluation_nfe_steps = (nfe,)
        model.viz_num_steps = nfe
        model.capture_latent_trajectories = False
        model.artifact_batch_limit = None
        observed_first_state: list[Any] = []
        wan_calls = [0]

        def capture(_module: Any, inputs: Any) -> None:
            if not observed_first_state:
                observed_first_state.append(inputs[0].detach().clone())

        def count(_module: Any, _inputs: Any, _output: Any) -> None:
            wan_calls[0] += 1

        pre_handle = model.forward_model.register_forward_pre_hook(capture)
        post_handle = model.forward_model.register_forward_hook(count)
        try:
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=True
            ):
                predicted = model.sample_future_deployable(
                    history,
                    actions,
                    morphology_index=morphology,
                    collect_artifacts=True,
                    sample_ids=sample_ids,
                )
        finally:
            pre_handle.remove()
            post_handle.remove()
        artifacts = model.pop_visualization_artifacts()
        counters = getattr(model, "_last_sampling_counters", None)
        latent_key = f"video_final_off_nfe_{nfe}"
        decoded_key = f"decoded_future_off_nfe_{nfe}"
        expected_noise = model._evaluation_noise(
            (1, 16, 4, 24, 120),
            device=device,
            dtype=torch.float32,
            base_seed=model.evaluation_noise_seed,
            sample_ids=sample_ids,
            stream=0,
            rank=0,
        )
        forbidden = {"video_clean", "tf_clean", "ground_truth_future_uint8"}
        if (
            not isinstance(artifacts, Mapping)
            or not isinstance(counters, Mapping)
            or forbidden.intersection(artifacts)
            or wan_calls[0] != nfe
            or counters.get("wan_calls_total") != nfe
            or counters.get("wan_calls_by_source_nfe")
            != {f"off:nfe_{nfe}": nfe}
            or counters.get("online_teacher_calls") != 0
            or counters.get("auxiliary_clean_available") != 0
            or counters.get("deployment_mode") != 1
            or len(observed_first_state) != 1
            or not torch.equal(observed_first_state[0], expected_noise)
            or latent_key not in artifacts
            or decoded_key not in artifacts
            or not torch.equal(artifacts[decoded_key], _to_uint8(predicted))
        ):
            raise HistoricalReferenceError(
                f"historical public sampler contract differs at NFE {nfe}"
            )
        results[str(nfe)] = {
            "video_latent": artifacts[latent_key].detach().cpu().contiguous(),
            "decoded_uint8": artifacts[decoded_key].detach().cpu().contiguous(),
            "video_initial_state_full_precision": (
                observed_first_state[0].detach().cpu().contiguous()
            ),
            "wan_calls": nfe,
            "online_teacher_or_feature_calls": 0,
            "future_rgb_sampler_input": False,
            "clean_video_latent_sampler_input": False,
        }
        total_wan_calls += nfe
    payload = {
        "schema_version": 1,
        "kind": "raw_physics_flow_historical_parent_reference",
        "implementation_commit": PARENT_COMMIT,
        "native_public_sampler": "sample_future_deployable",
        "condition_source": "off",
        "schedule_mode": "aligned",
        "nfe_grid": list(NFE_GRID),
        "strict_state_load": True,
        "historical_preserve_zero_support_attribute_absent": True,
        "total_wan_calls": total_wan_calls,
        "results": results,
        "validation_dataset_opened": False,
        "future_rgb_bytes_opened": False,
        "future_measured_state_opened": False,
        "protected_test_accessed": False,
    }
    try:
        torch.save(payload, partial)
        os.replace(partial, output)
    finally:
        if partial.exists():
            partial.unlink()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation-repo", type=Path, required=True)
    parser.add_argument("--parent-config", type=Path, required=True)
    parser.add_argument("--parent-snapshot", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wan-dir", type=Path, required=True)
    parser.add_argument("--videox-home", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return command_reference(build_parser().parse_args(argv))
    except HistoricalReferenceError as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
