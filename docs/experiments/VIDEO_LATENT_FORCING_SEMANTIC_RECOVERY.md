# Video Latent Forcing Semantic-Screen Recovery

Date frozen: 2026-08-01

This is an operational recovery amendment to the semantic predictor screen in
`VIDEO_LATENT_FORCING_POC_PROTOCOL.md`. It changes no scientific hypothesis,
dataset, split, target, seed, model size, optimizer, training budget, inference
NFE grid, metric, control, bootstrap rule, or pass threshold.

## Preserved v1 evidence

Execution `causal-vjepa2-semantic-screen-seed1234-20260801-v1` used scientific
commit `f144837279c80071d2f83327a217aff04863dae8`.

- Jobs `486721` and `486725` failed during Slurm setup and GPU-allocation
  validation, before scientific computation. Their corrected replacement,
  Stage A job `486729`, completed the real-checkpoint shape, repeatability, and
  prefix-causality checks.
- Stage B job `486731` completed the eight-rank interrupted/resumed mini-cache
  test. Its uninterrupted and resumed target bytes were identical.
- Stage C1 job `486744` completed the train-only PCA and full train/validation
  semantic caches. The PCA, train-target, and validation-target SHA-256 values
  were respectively
  `fc4173ef9e88725817ca8cc67c93b23d21dc9096927955483224db35518d0409`,
  `547c4579cf978ac2b9527cb038693259af678a2b07268ab1434706dc128051c4`, and
  `ed0ee6c76233b6072691854f767ce0e524867826b6d811c43988791f4011b034`.
  No protected-test RGB or semantic target was read or built.
- Job `486773` failed in external W&B environment sanitization before the
  calibration process started.
- Job `486793` reached exactly one finite calibration update, logged privately
  as W&B run `bs2ftsp8`, and then failed on the second backward pass. PyTorch
  reported unfinished DDP reduction for parameters 125 and 126, identified as
  `video_output_head.weight` and `video_output_head.bias`. The semantic loss
  consumed only the auxiliary output while the returned video output kept the
  unused video head inside DDP's forward graph.

The update-1 loss is excluded from scientific analysis. Execution v1 produced
no update-200 calibration checkpoint or completion record, no update-5,000
checkpoint, no validation predictions, and no gate decision. It therefore
contains no semantic quality observation that can influence this recovery.

## Code correction

The model forward now has a default-preserving `predict_video=True` option.
Semantic-only training and sampling set it to false, so the unused video head
is not evaluated and its placeholder is outside autograd. The default joint
forward, parameter names and shapes, state dict, parameter count
(`41,963,760`), auxiliary prediction, losses, and scientific schedule are
unchanged. Regression tests require auxiliary outputs to be bit-identical
between the default and semantic-only paths and require consecutive DDP
backward passes to leave every video-head gradient absent.

## Recovery execution contract

The corrected source is committed before execution and receives a new
immutable commit identity. A fresh execution ID ending in `-v2`, fresh artifact
root, fresh cache root, and fresh W&B run IDs are mandatory. No incomplete v1
calibration state is resumed or copied.

Although v1 Stages A, B, and C1 are valid systems evidence, their validators
bind them to the producing scientific commit. The recovery therefore reruns
Stages A, B, and C1 under the final corrected commit instead of weakening the
commit contract or mixing producer and consumer identities.

Before calibration, an additional real eight-rank B200 semantic-DDP preflight
must perform at least two consecutive optimizer iterations with the exact
model. It must verify finite synchronized loss/gradients, an unchanged video
head with absent video-head gradients, active auxiliary-head learning, the
expected state dict and parameter count, and all eight participating ranks.
The preflight is systems-only and cannot satisfy any scientific metric gate.

After those prerequisites pass, the recovery runs the unchanged 200-update
calibration, 5,000-update semantic training, full 890-clip by seven-NFE by
seven-control evaluation, and frozen semantic gate in order. Each downstream
stage remains fail-closed and may start only after an independent read-only
audit of its predecessor. The v1 tree and failed W&B run remain preserved as
part of the audit trail.
