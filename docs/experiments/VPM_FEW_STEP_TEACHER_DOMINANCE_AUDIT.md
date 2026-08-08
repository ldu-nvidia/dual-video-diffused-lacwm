# VPM few-step teacher-dominance audit

Date: 2026-08-08

## Decision question

Before spending a controlled run on Causal-rCM/Self-Forcing-style
distillation, does the existing causal VPM have a multi-step teacher that is
strictly better than its one-call endpoint on the same 64 validation clips?

This is a read-only secondary analysis of the completed frequency-forcing
study's video-only arm (`FPM`). It opens no protected test and trains nothing.
The source table is the complete `FPM` deployable evaluation grid:

```text
study: frequency-forcing-seed1234-20260808-8510b95-v3
arm/source: FPM/autonomous
rows.jsonl bytes: 1,948,552
rows.jsonl sha256: 5f7970660e9aa43f9ac54863121d2cec5543e82a1da04fc8753f288f845445ac
parent analysis identity: 32a9af9def2e0e2a86c8bd7a216b4c7b79603d073c8c03e7f639f7ca86e1088f
protected test accessed: false
```

## Fixed calculation

For each metric and candidate teacher NFE in `{2,4}`, pair by the registered
clip index and compute

\[
I_m = \frac{\overline{L}_{m,\,NFE1}-\overline{L}_{m,\,teacher}}
             {\overline{L}_{m,\,NFE1}}.
\]

Positive is better for the teacher. The interval is the percentile 95% paired
bootstrap interval from 10,000 resamples of 64 clips, seed 20260808. This is a
teacher-qualification audit, not a new model-selection endpoint.

## Results

| Teacher | Metric | NFE1 mean | Teacher mean | Relative improvement | Paired 95% interval |
|---|---|---:|---:|---:|---:|
| NFE2 | Wan-latent NMSE | 0.239028 | 0.239146 | -0.0494% | [-0.0682%, -0.0295%] |
| NFE2 | decoded MSE | 0.0189623 | 0.0189549 | +0.0391% | [+0.0119%, +0.0685%] |
| NFE2 | temporal MSE | 0.0132683 | 0.0132730 | -0.0357% | [-0.0716%, +0.00139%] |
| NFE4 | Wan-latent NMSE | 0.239028 | 0.290617 | -21.58% | [-24.07%, -19.30%] |
| NFE4 | decoded MSE | 0.0189623 | 0.0234617 | -23.73% | [-27.92%, -20.02%] |
| NFE4 | temporal MSE | 0.0132683 | 0.0156257 | -17.77% | [-20.48%, -15.34%] |

## Conclusion and claim boundary

The current multi-step sampler does **not** supply a superior teacher. NFE2
trades a statistically tiny decoded-MSE gain for worse latent and temporal
metrics; NFE4 is materially worse on every metric. Therefore an expensive
few-step-to-one-step distillation run is not justified yet: it would attempt to
distill a teacher that does not dominate the existing one-call student target.

This does not reject causal consistency/distillation as a direction. It fixes
the necessary next gate: first produce a causally deployable teacher that
improves all three paired validation metrics, then preregister distillation.
Until that gate passes, CAMP, intra-forward ordering, action-path repair, and a
train-only spectral physics critic are better uses of compute.
