# V-JEPA paired-latency validator recovery

This protocol recovers only the failed paired timing allocation `481133` for
the immutable study
`vjepa2-controlled-20260730-seed1234-9cf8e69-v3`.

Job `481133` completed runtime and study-input validation, created the intended
`paired_latency` directory, and then failed before model loading or timing
because the validator expected incomplete `{path, sha256}` records while the
producer correctly emitted `{path, sha256, bytes}`. Commit `9ba4073` fixes that
validator. The empty directory, original submission, job logs, and study tree
are retained unchanged.

## Scientific contract

Recovery repeats the preregistered comparison without changing its
computation:

- J1 autonomous deployable sampling at NFE 4;
- parameter-matched VPM autonomous deployable sampling at NFE 8;
- immutable test sample 0 and batch size 1;
- 20 warmup pairs and 100 timed pairs;
- J1-first on even rounds and VPM-first on odd rounds;
- both checkpoints resident in one process on one B200;
- synchronized timing around history preparation, Wan calls, and VAE decode;
- no future RGB, clean auxiliary target, online teacher, trajectory capture,
  or timed artifact materialization.

The scientific/model worktree stays clean at
`9cf8e6922f35a5d6645e3128545953723bf54da2`. Recovery tools run from a distinct
clean controller worktree descended from `9ba4073`. The controller proves the
inference-critical model trees and `tools/benchmark_vjepa2_inference.py` are
unchanged from the trained commit. The benchmark process puts the scientific
repository first for model imports, verifies every loaded `robot_wm` and `lam`
submodule after model/dataset instantiation and after timing, and records the
complete origin maps, their hashes, and the instantiated model/dataset classes.

## Provenance and output isolation

The read-only preflight requires:

- final array `481132` has exactly tasks 0–4, each `COMPLETED/0:0`;
- paired job `481133` is the exact `FAILED/2:0` one-B200 allocation with its
  original tokenized `SubmitLine`;
- the pinned stdout/stderr hashes and terminal validator error match;
- the original paired directory exists and is empty;
- the original planned analysis root is absent;
- study manifest and submission hashes/identities match the audited v3 bytes;
- the controller and scientific worktrees are canonical and clean;
- the Python executable resolves to the study-recorded interpreter;
- the study-recorded Wan directory contains the exact byte counts and SHA-256
  hashes pinned by the trained commit's B200 verifier for the transformer, VAE,
  config, null prompt, and null-prompt provenance;
- the study-recorded VideoX checkout is clean at exact commit `1d6d9c3`.

All new files are under:

```text
<study-parent>/_paired_recoveries/<study-id>/
  paired-481133-<controller7>-r1/
    protocol.json
    submission.json
    runtime.json
    logs/
    paired_latency/paired_j1_nfe4_vs_vpm_nfe8.json
    analysis/analysis.json
    analysis/analysis.md
```

Execution submits the allocation held, writes an exclusive receipt containing
the returned job ID and exact `sbatch` token vector, and only then releases it.
The allocation re-queries its own Slurm accounting and binds the actual
account, QOS, partition, CPU/GPU/memory/time request, work directory, node, job
ID, and SubmitLine in `runtime.json` before timing. The paired artifact embeds
records for the recovery protocol, receipt, runtime evidence, controller, and
scientific import origins. Wan content and VideoX Git state are validated
before model loading, immediately after timing, and again by the analyzer.
The analyzer accepts external paired evidence only with its matching recovery
receipt and writes only to the analysis paths bound by that receipt.

No live dependency is attached to the new job because Slurm has purged the
completed predecessor from controller state. The semantic `afterok` condition
is instead proven from all five terminal-success accounting rows and final
artifacts.

## Dry run and execution

Create a clean detached controller worktree at the final recovery commit, then
run:

```bash
BASE=/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train
CONTROLLER=$BASE/src/vjepa2-paired-recovery-<commit12>
STUDY=$BASE/runs/dual_video_diffusion/vjepa2_controlled_study/vjepa2-controlled-20260730-seed1234-9cf8e69-v3
SCIENTIFIC=$BASE/src/vjepa2-latent-forcing/dual-video-diffused-lacwm

LACWM_BASE=$BASE \
"$CONTROLLER/tools/slurm/submit_vjepa2_paired_latency_recovery.sh" \
  --study-root "$STUDY" \
  --controller-commit "<full-recovery-commit>" \
  --scientific-repo-root "$SCIENTIFIC"

# After reviewing the read-only preflight:
LACWM_BASE=$BASE \
"$CONTROLLER/tools/slurm/submit_vjepa2_paired_latency_recovery.sh" \
  --study-root "$STUDY" \
  --controller-commit "<full-recovery-commit>" \
  --scientific-repo-root "$SCIENTIFIC" \
  --execute
```

Defaults preserve `batch`, account `coreai_chef_posttrain`, QOS `normal`, one
B200, 32 CPUs, 256 GiB, and a four-hour limit. The launcher refuses alternate
values for this incident-specific recovery.

This recovered timing remains a diagnostic for the original J1@4 versus VPM@8
speed gate. It does not repair the scientific weakness that VPM quality was
already best around NFE 1–2, so even a faster J1@4 wall-clock result cannot by
itself establish a useful quality-matched acceleration claim.
