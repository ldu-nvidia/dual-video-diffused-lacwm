from __future__ import annotations

import numpy as np

from tools import video_residual_anchor_analyze as analysis


def test_paired_effect_detects_uniform_improvement() -> None:
    baseline = np.linspace(1.0, 2.0, 64)
    candidate = baseline * 0.9
    effect = analysis._paired_effect(baseline, candidate, seed=7)
    assert abs(effect["relative_improvement"] - 0.1) < 1e-12
    assert effect["one_sided_bonferroni_lower_bound"] > 0.09
    assert effect["n"] == 64


def test_bonferroni_family_is_frozen_to_twelve_contrasts() -> None:
    assert analysis.BONFERRONI_CONTRASTS == 12
    assert analysis.ONE_SIDED_ALPHA == 0.05 / 12
    assert analysis.BOOTSTRAP_REPLICATES == 10_000
