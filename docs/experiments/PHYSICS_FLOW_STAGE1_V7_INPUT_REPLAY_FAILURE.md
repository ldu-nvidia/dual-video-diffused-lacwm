# Raw physics-flow Stage-1 v7 input-replay qualification failure

Date: 2026-08-09

Status: **v7 remains failed; zero v7 endpoint rows; v8 may evaluate its frozen
checkpoints only as a new single-seed exploratory study**

## Terminal v7 fact

The prospective v7 chain at source
`19717d33a70c344629deff968c2b2309f7083005` completed both 200-update
training arms, then evaluation job 507846 stopped before parent parity,
endpoint materialization, scoring, analysis, or audit. The terminal v7 decision
is permanently `STOP_EXACT_REPAIR_EQUIVALENCE`; later work must not relabel v7
as passing.

The canonical immutable inputs are:

- study: `.../raw_physics_flow_stage1/raw-physics-flow-stage1-20260808-19717d3-v7`;
- cache: `.../raw_physics_flow_cache/raw-physics-flow-cache-20260808-19717d3-v7`;
- log: `.../logs/dual_video_diffusion/raw-physics-flow-stage1-20260808-19717d3-v7/eval-507846.out`;
- `FLOW-OFF` snapshot: 4,248,290,554 bytes, SHA-256
  `7e754497526da41a812ad89d7f76e5043a19b0dbf15bbe24686f35494498c21c`;
- `RAW-FLOW` snapshot: 4,248,290,554 bytes, SHA-256
  `d37093b7877c8893a557cc806fd40e312377ec66de19f0b8435d659f43ab298f`.

The v7 study has no `training_pairing.json`, parent-parity receipt,
`evaluation/`, scoring output, or `analysis/`. Therefore v7 produced no video
quality effect estimate.

## Why exact-output qualification was invalid

The v7 gate required a fresh B200 rerun to reproduce v5 floating-point outputs
and trained parameters bit for bit. The rerun did not preregister deterministic
CUDA algorithms, fixed GEMM/reduction workspaces, or a validated deterministic
multi-seed protocol. It is therefore valid to require exact causal inputs,
event structure, frozen tensors, and finite outputs; it is not valid to turn
an unregistered numerical tolerance into a post-hoc pass criterion.

Read-only inspection establishes the narrower facts that v8 now checks
prospectively:

- the eight train/validation cache arrays are byte-identical to v5;
- five all-rank causal-input hashes at each of 200 updates in each arm give
  exactly 2,000 hash comparisons;
- all 23 input, hyperparameter, probe, clock, index, and order metrics replay
  exactly at every update, while nine output/telemetry diagnostics are finite
  and their drift is reported descriptively;
- both snapshots have the same 1,686-tensor schema as v5 and all tensors are
  finite;
- `FLOW-OFF` differs only in the 495 registered trainable tensors (action
  encoder/pool/control projection, morphology token, and 480 LoRA tensors);
- `RAW-FLOW` differs in those 495 plus the five physics-flow adapter tensors;
  every remaining frozen parameter and buffer is bit-identical.

These observations explain why the exact v7 rerun gate stopped. They do not
establish that the treatment helps, nor do they erase the v7 failure.

## Prospective v8 boundary

V8 is a fresh registration and output namespace that reuses the immutable v7
checkpoints, traces, configs, and cache arrays without retraining and without a
new W&B run. Before any v8 video is generated, registration must seal the exact
causal-input replay receipt and record finite output/trainable drift with no
numeric acceptance threshold. It then applies the already-frozen v7 endpoint
grid, causal boundary, metrics, and decision gates.

Any v8 result is a single-seed exploratory result about these frozen
checkpoints. A confirmatory claim requires a future, prospectively
deterministic, multi-seed training and evaluation study. V5 remains input
lineage evidence only, and neither v5 nor v7 endpoint outcomes are imported.
