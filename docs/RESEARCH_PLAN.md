# Phased implementation and research plan

Each phase has a falsifiable gate.  A failed gate is a result; it is not a
reason to scale compute.

## Phase 0 — Bootstrap and provenance

Deliverables:

- production LACWM history and run identity pinned;
- Latent Forcing mapping and notices recorded;
- deterministic per-view TF transforms;
- dual corruption, schedule, adapter and head contract tests;
- no weights, datasets, logs, credentials or generated artifacts in Git.

Gate: clean repository; all focused tests and CPU smoke pass; no-op modules are
exactly zero; ordinary Git push succeeds.

## Phase 1 — Reproduce a causal baseline

Add an explicit-action evaluation entry point that accepts history, proposed
actions and output shape without reading future RGB.  Establish fixed
episode-level train/validation/test splits and measure the production checkpoint
at 4, 8, 12 and 20 Wan NFEs.

Required checks:

- dual-off fixed-seed output matches the baseline;
- changing hidden future RGB cannot change a generated sample;
- changing planned actions measurably changes predictions;
- batch-1 B200 p50/p95 latency and peak memory are recorded;
- metrics are computed per camera view, not across the stacked canvas.

Gate: quality within 1% and latency within 5% of the pinned baseline, with no
future-RGB dependence.

## Phase 2 — Qualify TF representations

Compare complete causal RFFT, localized STFT, temporal Haar/wavelets,
magnitude-only, high-pass-only, and per-frame-VAE-latent transforms.  Normalize
physical frame intervals across datasets before assigning frequency meaning.
Estimate whitening statistics from the training split only.

Required checks:

- complete RFFT round-trip error near numerical tolerance;
- known sinusoid and phase-shift response;
- no camera-boundary mixing;
- window-validity and future/history masks;
- frame shuffling and static-video controls;
- Wan VAE reconstruction ceiling measured separately.

Gate: deterministic transform, stable held-out normalization, correct phase
response, and a measurable ability to distinguish coherent motion.

## Phase 3 — Integrate dual flow into Wan

Expose Wan hidden tokens behind a tested compatibility wrapper.  Add a TF token
adapter, TF clock embedding, TF velocity head, masked TF loss and joint Euler
state update.  Keep the pretrained 48-channel input and video head unchanged.

Required schedule coverage:

- aligned;
- independent;
- strict TF-first cascade;
- shifted/overlapping TF-first;
- reversed video-first.

Gate: dual-off baseline identity; old checkpoints load; both branches have
finite nonzero gradients when enabled; invalid/history losses are zero; a
two-update one-B200 smoke and three accumulated 8xB200 updates resume exactly.

## Phase 4 — Controlled DROID study

Use fixed splits and at least three seeds for core comparisons:

1. causal explicit-action single-stream LACWM;
2. parameter-matched single-stream control;
3. aligned dual clocks;
4. TF-first cascade;
5. shifted TF-first;
6. reversed video-first;
7. random auxiliary state;
8. magnitude-only versus real/imag phase;
9. RFFT versus STFT/wavelet;
10. raw-video versus per-frame-VAE TF targets.

Match samples, optimizer updates, global batch, LoRA rank, seeds and wall-clock
accounting.  Report extra parameters, FLOPs, memory and GPU-hours.

Go gate: at 8 NFE, dual diffusion must beat the 8-NFE single-stream model on a
preregistered temporal metric with paired uncertainty, survive
parameter-matched/random-state controls, and approach the 20-NFE baseline
within 5% quality without more than roughly 10% end-to-end latency overhead.

## Phase 5 — Multi-dataset B200 scaling

Progress from one B200 to one 8xB200 node, full DROID, then the production
four-dataset mixture.  Preserve the effective global batch initially and reuse
the production DDP, preflight, durable Slurm checkpoint and exact-resume
infrastructure.

Gate: at least 70% useful scaling, no nonfinite loss, no input starvation,
deterministic resume, and held-out gains across datasets, robots and views.

## Phase 6 — Low-NFE schedule optimization

Evaluate 2, 4, 6, 8, 12 and 20 true Wan NFEs.  Charge Heun twice per interval.
Tune schedule allocation on validation only, freeze it, and evaluate test once.
Treat trajectory distillation or consistency training as separate interventions.

Report complete latency: action encoding, Wan calls, TF updates, VAE decoding,
transfers and synchronization.  A 5--10 Hz control loop permits approximately
200--100 ms per complete decision rollout, not merely per decoded frame.

## Phase 7 — Physical-AI/DAgger validation

Evaluate action-conditioned counterfactual prediction, candidate-action ranking,
planner regret, rollout calibration, simulator task success/collisions, DAgger
expert-label efficiency and closed-loop latency.  Every rollout must use only
observed history and proposed future actions.

## Metrics

Quality: PSNR, SSIM, LPIPS, frame FID, and appropriately sampled FVD/KVD, plus
Wan-VAE-reconstructed ground truth as an upper bound.

Temporal: motion-weighted temporal LPIPS, optical-flow warp error, flicker,
temporal log-spectrum error, complex-coefficient error, circular phase-increment
error and phase-locking-value error.

Conditioning: action recovery, same-history/different-action sensitivity,
per-morphology/dataset/view breakdowns and missing-view robustness.

Efficiency: Wan NFE, FLOPs, total/trainable parameters, peak memory, samples and
GPU-hours to quality threshold, batch-1 B200 p50/p95 latency, decoded frames/s
and complete first-rollout latency.
