# Raw physics-flow Stage-1 `e632344-v5` failure seal

Date: 2026-08-09

Status: **engineering chain completed through paired training; evaluation failed
before endpoint materialization; zero scientific quality result**

## Outcome

The prospective `e632344-v5` chain is not a positive or negative result about
raw physics-flow conditioning. Registration, both causal-flow cache builds,
both cache audits, the historical-parent seal, the matched `FLOW-OFF` and
`RAW-FLOW` training arms, and native-parent sampler parity completed. Evaluation
job 507701 then stopped on a deterministic configuration mismatch before it
wrote an endpoint row. The created `evaluation/` directory is empty. There is
no `endpoint_rows.jsonl`, `analysis.json`, `audit.json`, or `run_complete.json`.

Accordingly, this chain produced:

- **0 endpoint rows**;
- **0 scored videos**;
- **0 bootstrap comparisons**;
- **0 Stage-1 gate decision**; and
- **no evidence for or against a video-quality benefit**.

It must not be summarized as a model failure, a model improvement, or an
estimate of effect size.

## Exact immutable chain

The source checkout was clean at commit
`e632344a8f031a520640779d130d805aa1dd0b6e` and tree
`e5c983e8778cde045ecf0d53218d253574fb3aa8`. The canonical remote paths are:

- source: `.../src/worktrees/raw-physics-flow-stage1-e632344`;
- cache: `.../raw_physics_flow_cache/raw-physics-flow-cache-20260808-e632344-v5`;
- study: `.../raw_physics_flow_stage1/raw-physics-flow-stage1-20260808-e632344-v5`;
- logs: `.../logs/dual_video_diffusion/raw-physics-flow-stage1-20260808-e632344-v5`.

The terminal Slurm records were:

| Job | Stage | State | Exit | Elapsed | Allocation |
|---:|---|---|---:|---:|---|
| 507695 | register | COMPLETED | 0:0 | 00:01:00 | 1 B200, 32 CPU, 256 GiB |
| 507696 | train cache | COMPLETED | 0:0 | 00:02:03 | 1 B200, 32 CPU, 256 GiB |
| 507697 | validation cache | COMPLETED | 0:0 | 00:00:33 | 1 B200, 32 CPU, 256 GiB |
| 507698 | audit and seal | COMPLETED | 0:0 | 00:00:48 | 1 B200, 32 CPU, 256 GiB |
| 507699 | `FLOW-OFF` training | COMPLETED | 0:0 | 00:11:46 | 8 B200, 160 CPU, 1000 GiB |
| 507700 | `RAW-FLOW` training | COMPLETED | 0:0 | 00:11:49 | 8 B200, 160 CPU, 1000 GiB |
| 507701 | evaluation | **FAILED** | 1:0 | 00:10:52 | 8 B200, 160 CPU, 1000 GiB |

## What did complete successfully

The train cache contains 512 clips, including 415 D405 clips; the validation
cache contains 64 clips, including 48 D405 clips. Each audit rehashed four
arrays, reconstructed the three deterministic controls with zero error,
verified 2,048 train plus 256 validation row-tensor hashes, and replayed 512
train plus 64 validation lineage identities. These facts establish internal
cache consistency under the v5 implementation, subject to the strict-access
caveat below.

Both arms completed all 200 registered updates. The sealed pairing receipt has
SHA-256 `4776b2792e1aa30d95566c9e6fade3ccadc7622053319051b1186124ac174e37`
and identity
`37413d615de5c5e8b4f89ada0e086ed70e33d8822ee58852ddae9f6b3361de9c`.
It verifies 1,000 exact all-rank comparisons across clip indices, actions,
flow tensors, video noise, and timesteps. The two traces each contain 203 rows
and both training completions report update 200. The uncopied large snapshots
remain pinned by the pairing receipt:

| Arm | Snapshot bytes | Snapshot SHA-256 |
|---|---:|---|
| `FLOW-OFF` | 4,248,290,490 | `d59c06d44bc4989a072061993452a4f784f915d18af73cd29d067a3f548ba3da` |
| `RAW-FLOW` | 4,248,290,490 | `870d17ede7ec60c513ebbdaf43dccfb55012107f3caf28c6edbfea4555923e6f` |

The native-parent parity receipt has SHA-256
`868e98dd851efd3557ed514139f26f6d0e40a6ff98e69102b2abab96c16996b0`
and identity
`99cfe4104296d4ff7a40be8e09727d267c2755f8a0b41dc8399e29222bb0d451`.
It records bitwise-equal direct, current-adapter, and isolated historical
parent latents and decoded outputs at NFE 1, 2, and 4. This validates the parent
adapter seam, not the unexecuted treatment comparison.

## Evaluation blocker

Both resolved training configs inherited
`model.dual_diffusion.evaluation_noise_seed: 20260726`. The exact untouched
parent contract requires `20260729`. Evaluation loaded both 4.25 GB arm
snapshots and the native parent, then the registered cross-model seed gate in
`command_evaluate` observed the unequal set and raised:

```text
training arms and native parent evaluation noise base seeds differ
```

This occurred before construction of the LPIPS scorer, sampler hooks, endpoint
loop, target materialization, or metric rows. The repair is to register
`20260729` for both training-arm model configs and validate that contract before
expensive model loading. A seed-only replay is prohibited: the next claimable
chain must also repair strict observed-state access and rerun end to end under
fresh namespaces.

## Independent strict-access caveat

The v5 cache builder forms the numeric renderer state only from frame 4:

```python
state["joint_states"][frame4]
state["gripper_states"][frame4]
```

However, `NpzFile.__getitem__` materializes the complete `.npy` member before
Python applies the row index. Thus the v5 receipt field
`future_measured_state_opened: false` is not literally valid at the
application-byte-access boundary. The future measured-state values do not
directly enter the renderer tensor: only the selected frame-4 values do, and
future pose endpoints come from the separately registered immutable action
array. That supports only the narrower statement that future measured-state
**values were not used numerically**. It does not support the stronger claim
that future measured-state **bytes were never opened**. Whole-file stat
metadata also entered v5 provenance identities.

This caveat is independent of the seed mismatch. Even if job 507701 had passed,
v5 would have remained provisional until a strict observed-row reader and fresh
cache replay established the intended byte-level causal boundary. Existing v5
receipts are preserved unchanged; their false field is documented, not
rewritten.

## Fresh-chain requirements

The next chain must combine both repairs and use fresh source, cache, study,
log, training, evaluation, and W&B namespaces. Before validation opens, its
preregistered gates must require:

1. exact-range reads of only frame-4 joint and gripper payload bytes from the
   real ZIP_STORED NPZ layout, rejection of compressed/unsupported layouts
   before payload access, truthful range receipts, and no measured action read
   from the episode state archive;
2. cache/provenance schema bumps plus future-state mutation invariance and
   frame-4 sensitivity tests;
3. `evaluation_noise_seed == 20260729` in both arm configs and runtime models,
   checked before expensive loading;
4. byte-identical v6 versus sealed-v5 `raw`, `episode_shuffled`,
   `timeshift_plus_one`, and `hold_current` flow arrays, plus equal row numeric
   hashes while provenance identities are expected to differ; and
5. numerically or bitwise equivalent v6 versus v5 paired training traces and
   weights, excluding only preregistered metadata/config-identity differences.

Any mismatch stops before endpoint scoring. Only a fresh, fully sealed
registration-to-evaluation replay can yield a scientific Stage-1 conclusion.

## Evidence mirror

Selected immutable receipts and logs were copied without modification to:

`/mnt/data1/ldu/research/Dual Video Diffusion/artifacts/raw_physics_flow_stage1/raw-physics-flow-stage1-20260808-e632344-v5/remote`

The mirror excludes generated flow arrays, W&B binaries, the 15.8 MB historical
reference bundle, and both 4.25 GB snapshots. All copied-file hashes were
computed independently on the remote canonical files and the local mirror and
matched. The complete machine-checkable list is in
`PHYSICS_FLOW_STAGE1_V5_FAILURE_MANIFEST.sha256`.
