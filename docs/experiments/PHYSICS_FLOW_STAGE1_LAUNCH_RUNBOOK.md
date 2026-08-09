# Raw physics-flow Stage-1 frozen-v7 exploratory evaluation runbook

Status: **prepared only; do not execute or launch from this source checkout.** A launch
requires an independently acknowledged exact 40-character v8 source commit and
a fresh commit-derived source, study, and log namespace.

This v8 chain performs no cache build, training, checkpoint write, or W&B run.
It first registers and seals an honest causal-input replay of immutable v7
against the v5 input-lineage diagnostic, then evaluates the frozen v7
checkpoints under the endpoint grid and decision gates that were frozen before
v7 outcomes. V7 remains `STOP_EXACT_REPAIR_EQUIVALENCE`; v8 cannot
retroactively pass it.

## 1. Immutable inputs and fresh paths

Run on `gcp-nrt-login-002` only after source audit acknowledgment. Replace the
single placeholder; never infer it from a moving branch.

```bash
set -euo pipefail
umask 077

BASE=/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train
BRANCH=research/raw-physics-flow-stage1
EXPECTED_COMMIT=REPLACE_WITH_AUDITOR_ACKNOWLEDGED_40_CHARACTER_COMMIT
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]]
SHORT=${EXPECTED_COMMIT:0:7}

SOURCE_MIRROR=$BASE/src/dual-video-diffused-lacwm
SOURCE_REPO=$BASE/src/worktrees/raw-physics-flow-stage1-$SHORT-v8
PYTHON_BIN=$BASE/envs/lacwm-b200-py310/bin/python
WAN_DIR=$BASE/wan_fun_1.3b_control
VIDEOX_HOME=$BASE/VideoX-Fun-1d6d9c3

V5_STUDY_ROOT=$BASE/artifacts/dual_video_diffusion/raw_physics_flow_stage1/raw-physics-flow-stage1-20260808-e632344-v5
V7_CACHE_ROOT=$BASE/artifacts/dual_video_diffusion/raw_physics_flow_cache/raw-physics-flow-cache-20260808-19717d3-v7
V7_STUDY_ROOT=$BASE/artifacts/dual_video_diffusion/raw_physics_flow_stage1/raw-physics-flow-stage1-20260808-19717d3-v7
V7_LOG_ROOT=$BASE/logs/dual_video_diffusion/raw-physics-flow-stage1-20260808-19717d3-v7

STUDY_ROOT=$BASE/artifacts/dual_video_diffusion/raw_physics_flow_stage1/raw-physics-flow-stage1-20260809-$SHORT-v8-exploratory
LOG_ROOT=$BASE/logs/dual_video_diffusion/raw-physics-flow-stage1-20260809-$SHORT-v8-exploratory
REGISTRATION=$STUDY_ROOT/registration.json

test -d "$V5_STUDY_ROOT"
test -d "$V7_CACHE_ROOT"
test -d "$V7_STUDY_ROOT"
test "$(sha256sum "$V7_STUDY_ROOT/registration.json" | awk '{print $1}')" = 82c6b16c59454835be90889451bfd6c365c6e93499c5861d206dc2791a37e786
test "$(sha256sum "$V7_CACHE_ROOT/cache_registration.json" | awk '{print $1}')" = a1af8831819455207b5b9a47fe33b2da5a50cbd9d21c24b96646e553a4f8c0a8
test "$(sha256sum "$V7_LOG_ROOT/eval-507846.out" | awk '{print $1}')" = 9aea10af872bfa81389553da9a2d8045cf7cf0ff1ebbc849d8d9184220263d44
test ! -e "$V7_STUDY_ROOT/training_pairing.json"
test ! -e "$V7_STUDY_ROOT/parent_sampler_parity.json"
test ! -e "$V7_STUDY_ROOT/evaluation"
test ! -e "$V7_STUDY_ROOT/analysis"
test ! -e "$SOURCE_REPO"
test ! -e "$STUDY_ROOT"
test ! -e "$LOG_ROOT"
```

The registration code additionally binds the exact v7/v5 registrations,
traces, completions, configs, checkpoints, all eight cache arrays, and the v7
failure log. It verifies 2,000 exact causal-input hashes, 9,200 exact
input/probe/clock/index/order values, 7,200 finite output-diagnostic values,
the exact 1,686-tensor model schema, the registered 495/500 trainable mismatch
families, and bit identity of every other frozen parameter and buffer. Finite
trainable/output drift is descriptive; no numerical pass threshold exists.

## 2. Freeze the audited v8 source

```bash
git -C "$SOURCE_MIRROR" fetch --no-tags origin "refs/heads/$BRANCH"
test "$(git -C "$SOURCE_MIRROR" rev-parse FETCH_HEAD)" = "$EXPECTED_COMMIT"
git -C "$SOURCE_MIRROR" worktree add --detach "$SOURCE_REPO" "$EXPECTED_COMMIT"
test "$(git -C "$SOURCE_REPO" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SOURCE_REPO" status --porcelain --untracked-files=all)"
mkdir -p "$(dirname "$STUDY_ROOT")" "$LOG_ROOT"
```

## 3. Submit registration then evaluation

The registration allocation opens no generated-video outcome. It writes a new
registration plus `v7_v5_causal_input_replay_gate.json`; it does not write to
v5 or v7. The sole evaluation allocation revalidates that gate, runs native
parent parity before validation opens, materializes the frozen endpoint grid,
scores, analyzes, and audits. `afterok` prevents evaluation after any
registration failure.

```bash
BASH_PREFIX="/bin/bash -lc 'set -euo pipefail; umask 077; export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 LACWM_PYTHON=$PYTHON_BIN WAN_DIR=$WAN_DIR VIDEOX_HOME=$VIDEOX_HOME WANDB_MODE=disabled; source $SOURCE_REPO/tools/env/activate_b200.sh; cd $SOURCE_REPO;"

REGISTER_JOB=$(sbatch --parsable \
  --job-name=pf-v8-register-$SHORT \
  --output="$LOG_ROOT/register-%j.out" \
  --nodes=1 --ntasks=1 --gpus-per-node=1 --cpus-per-task=32 \
  --mem=256G --time=02:00:00 \
  --partition=batch --account=coreai_chef_posttrain --qos=short --no-requeue \
  --exclude=pool0-0081,pool0-0089 \
  --wrap="$BASH_PREFIX \
    $PYTHON_BIN tools/physics_flow_stage1.py register-frozen-v7-evaluation \
      --output $STUDY_ROOT --source-repo $SOURCE_REPO \
      --expected-commit $EXPECTED_COMMIT \
      --v7-cache-root $V7_CACHE_ROOT --v7-study-root $V7_STUDY_ROOT \
      --v5-reference-study-root $V5_STUDY_ROOT; \
    $PYTHON_BIN tools/physics_flow_stage1_workflow.py plan \
      --registration $REGISTRATION'")

EVAL_JOB=$(sbatch --parsable --dependency=afterok:$REGISTER_JOB \
  --job-name=pf-v8-eval-$SHORT --output="$LOG_ROOT/eval-%j.out" \
  "$SOURCE_REPO/tools/slurm/physics_flow_stage1.sbatch" \
  --mode evaluate --registration "$REGISTRATION" \
  --repo-root "$SOURCE_REPO" --expected-commit "$EXPECTED_COMMIT" \
  --python "$PYTHON_BIN")

printf 'register=%s eval=%s\n' "$REGISTER_JOB" "$EVAL_JOB"
```

No command above sets an online W&B entity/project, creates a v8 training
directory, or invokes `write-arm-plan`. The evaluation entrypoint explicitly
sets `WANDB_MODE=disabled` and points model loading at the immutable v7 run
directories.

## 4. Read-only monitoring and terminal evidence

```bash
squeue -j "$REGISTER_JOB,$EVAL_JOB" -o '%.18i %.30j %.2t %.10M %.10l %R'
sacct -j "$REGISTER_JOB,$EVAL_JOB" \
  --format=JobID,JobName%32,State,ExitCode,Elapsed,Start,End
tail -n 80 "$LOG_ROOT/register-$REGISTER_JOB.out"
tail -n 120 "$LOG_ROOT/eval-$EVAL_JOB.out"
```

Terminal success requires nonempty immutable files at:

```bash
test -s "$STUDY_ROOT/v7_v5_causal_input_replay_gate.json"
test -s "$STUDY_ROOT/parent_sampler_parity.json"
test -s "$STUDY_ROOT/evaluation/inventory.json"
test -s "$STUDY_ROOT/analysis/analysis.json"
test -s "$STUDY_ROOT/analysis/audit.json"
```

Report the preregistered decision and all effect/interval tables, including
negative results. Interpret either `ADVANCE_RAW_FLOW_SCAFFOLD` or
`STOP_FIXED_RAW_FLOW` only within the registered single-seed exploratory claim
boundary. A confirmatory video-quality claim remains deferred to a future
prospectively deterministic, multi-seed study.
