# V-JEPA native-all-video post-study protocol

Date frozen: 2026-07-30, before update-1000 native-all-video scoring.

## Question

Does the faithful V-JEPA training intervention improve the LACWM video
denoiser when every inference call advances the native Wan video schedule, and
does enabling the autonomously generated V-JEPA state add a material
same-checkpoint inference benefit?

This is a warm-started ABC robotics-video study. It does not establish
from-scratch convergence, perceptual quality, other datasets, or a general
V-JEPA result.

## Runtime intervention

The final immutable update-1000 VPM, A1, and J1 resolved configurations and
snapshots are strict-loaded. Model and dataset code are unchanged.

- VPM/off: native all-video schedule, auxiliary state and clock injection off.
- A1/off: native all-video schedule, auxiliary state and clock injection off.
- J1/aligned: native all-video schedule, generated auxiliary state and clock
  aligned to the current video clock and injected.
- J1/off: the same J1 checkpoint and schedule with state and clock injection
  disabled.

The fixed NFE grid is `1,2,4,6,8,12,20`. Validation uses all 64 immutable
clips; a separately registered, eligible lockbox uses 128 episode-disjoint
clips and the same frozen grid. No NFE is selected on either split.

All sampling is history-only through `sample_future_deployable`. Clean future
RGB, clean future V-JEPA targets, oracle sources, and an online teacher are
forbidden.

## Fixed K=4 composite gate

Lockbox entry requires all three paired K=4 comparisons to pass
`tools.vjepa2_nfe_frontier.same_nfe_attribution_gate`:

1. A1/off versus VPM/off: training-objective effect.
2. J1/aligned versus J1/off: generated-feature-use effect.
3. J1/aligned versus VPM/off: end-to-end effect.

For each comparison, temporal-difference MSE relative-improvement CI-low must
be at least 3%, while video-latent NMSE and decoded-MSE CI-lows must both be
above -1%. The protocol uses 10,000 paired clip bootstraps, 95% intervals, and
seed 1234. All three components and the endpoint audit are conjunctive; a tiny
or isolated effect cannot be reported as an obvious advantage.

Aligned-versus-off isolates whether enabling the generated auxiliary feedback
helps. It does not by itself prove use of sample-specific semantic content; a
same-checkpoint aligned-versus-shuffled contrast would be required for that
stronger claim.

## Endpoint and execution audit

On validation, every arm additionally compares native-off at K=`1,2,4` with
cascade-off at total calls `2K`. The evaluator requires exact hashes for the
sampler-exposed float16 video endpoint and uint8 decoded endpoint, as well as
identical initial video noise, auxiliary noise, reference latents, and native
video-phase sigma nodes.

Runtime wrappers count actual:

- Wan backbone calls;
- video scheduler updates;
- auxiliary Euler invocations; and
- nonzero auxiliary sigma transitions.

Native K must record `(K,K,K,K)`. Cascade 2K must record
`(2K,K,2K,K)`. Online-teacher calls must remain zero.

## Validation-first lockbox rule

The lockbox registration path is not opened or stat'ed until an identity-valid
validation report reproduces the complete three-way K=4 gate, endpoint audit,
training/evaluator commits, and final VPM/A1/J1 artifact identities. Lockbox
scoring cannot reselect an NFE or change the gate.

The Slurm launcher is dry-run-first and requires explicit account, QOS,
partition, scientific output directory, and log directory. It uses one node,
eight B200 GPUs, and eight distributed ranks. No job is launched by adding
this protocol or its tooling.
