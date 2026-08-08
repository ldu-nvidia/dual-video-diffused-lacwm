#!/usr/bin/env python3
"""Post-hoc per-horizon and motion-magnitude diagnostics for dense Stage-0.

This script is descriptive only. It reads sealed predictions and targets and does
not alter the preregistered decision or write into the run directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CONDITIONS = (
    "history_only",
    "history_plus_aligned_action",
    "history_plus_episode_shuffled_action",
    "history_plus_zero_action",
)


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    difference = prediction - target
    mse = float(np.mean(difference * difference))
    epe = float(np.mean(np.sqrt(np.sum(difference * difference, axis=1))))
    pred_flat = prediction.reshape(len(prediction), -1)
    target_flat = target.reshape(len(target), -1)
    cosine = float(
        np.mean(
            np.sum(pred_flat * target_flat, axis=1)
            / (
                np.linalg.norm(pred_flat, axis=1)
                * np.linalg.norm(target_flat, axis=1)
                + 1e-12
            )
        )
    )
    return {"dense_mse": mse, "endpoint_error": epe, "directional_cosine": cosine}


def effect(reference: dict[str, float], candidate: dict[str, float]) -> dict[str, float]:
    return {
        "dense_mse_relative_improvement_percent": 100.0
        * (1.0 - candidate["dense_mse"] / reference["dense_mse"]),
        "endpoint_error_relative_improvement_percent": 100.0
        * (1.0 - candidate["endpoint_error"] / reference["endpoint_error"]),
        "directional_cosine_absolute_gain": candidate["directional_cosine"]
        - reference["directional_cosine"],
    }


def summarize(predictions: dict[str, np.ndarray], target: np.ndarray) -> dict:
    condition_metrics = {name: metrics(value, target) for name, value in predictions.items()}
    return {
        "target_mean_endpoint_magnitude": float(
            np.mean(np.sqrt(np.sum(target * target, axis=1)))
        ),
        "metrics": condition_metrics,
        "aligned_vs_history_only": effect(
            condition_metrics["history_only"],
            condition_metrics["history_plus_aligned_action"],
        ),
        "aligned_vs_episode_shuffled_same_model": effect(
            condition_metrics["history_plus_episode_shuffled_action"],
            condition_metrics["history_plus_aligned_action"],
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()

    derived_path = args.run_root / "derived_dense_features_targets.npz"
    model_path = args.run_root / "model_state_and_predictions.npz"
    with np.load(derived_path) as derived, np.load(model_path) as model:
        target = derived["val_future_dense_flow_target"].reshape(64, 8, 2, 12, 20)
        predictions = {
            name: model[f"prediction_{name}"].reshape(64, 8, 2, 12, 20)
            for name in CONDITIONS
        }

    per_horizon = []
    for horizon in range(8):
        per_horizon.append(
            {
                "horizon_transition": f"{4 + horizon}->{5 + horizon}",
                **summarize(
                    {name: value[:, horizon] for name, value in predictions.items()},
                    target[:, horizon],
                ),
            }
        )

    # A unit is one validation clip at one future horizon. Bins are determined
    # only from the clean scoring target and are explicitly post-hoc diagnostics.
    unit_magnitude = np.mean(np.sqrt(np.sum(target * target, axis=2)), axis=(2, 3))
    thresholds = np.quantile(unit_magnitude.reshape(-1), [0.25, 0.5, 0.75])
    bin_index = np.digitize(unit_magnitude, thresholds, right=True)
    magnitude_bins = []
    labels = ("q1_low", "q2", "q3", "q4_high")
    for index, label in enumerate(labels):
        mask = bin_index == index
        selected_target = target[mask]
        selected_predictions = {name: value[mask] for name, value in predictions.items()}
        magnitude_bins.append(
            {
                "bin": label,
                "clip_horizon_units": int(mask.sum()),
                **summarize(selected_predictions, selected_target),
            }
        )

    payload = {
        "schema_version": 1,
        "kind": "dense_top_flow_stage0_posthoc_diagnostic",
        "post_hoc": True,
        "registered_decision_unchanged": "NO_GO",
        "unit_of_motion_magnitude_stratification": "validation clip x future horizon",
        "motion_magnitude_quartile_thresholds": [float(value) for value in thresholds],
        "per_horizon": per_horizon,
        "motion_magnitude_bins": magnitude_bins,
        "limitations": [
            "descriptive post-hoc slices with no multiplicity-adjusted inference",
            "motion magnitude does not identify contact, robot, object, or background pixels",
            "clip-horizon units from the same episode are correlated",
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
