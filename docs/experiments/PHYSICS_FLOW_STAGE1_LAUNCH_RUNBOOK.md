# Raw physics-flow Stage-1 launch runbook

Status: **prepared only; do not execute any command in this document until an
independent source auditor explicitly acknowledges the exact 40-character
`EXPECTED_COMMIT`.** Preparing this runbook did not create a cluster checkout,
cache, study registration, training run, evaluation output, or validation
outcome.

The chain is fail-closed and uses fresh source, cache, study, log, and training
paths. `afterok` dependencies stop every downstream job if registration, either
cache, either audit, parent-lineage sealing, either matched training arm, or
historical-parent parity fails. The evaluation job itself performs trace
comparison, the isolated historical 6560866 parity gate, all 21 target-blind
endpoint materializations, scoring, analysis, and final replay audit in that
order.

## 1. Exact immutable inputs

Run on `gcp-nrt-login-002` only after audit acknowledgment. Replace the single
placeholder with the auditor-approved commit; do not infer it from a moving
branch.

```bash
set -euo pipefail
umask 077

BASE=/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train
BRANCH=research/raw-physics-flow-stage1
EXPECTED_COMMIT=REPLACE_WITH_AUDITOR_ACKNOWLEDGED_40_CHARACTER_COMMIT
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]]
SHORT=${EXPECTED_COMMIT:0:7}

SOURCE_MIRROR=$BASE/src/dual-video-diffused-lacwm
SOURCE_REPO=$BASE/src/worktrees/raw-physics-flow-stage1-$SHORT
PARENT_SOURCE_REPO=$BASE/src/vjepa2-faithful-cascade-656086686dae/dual-video-diffused-lacwm
PYTHON_BIN=$BASE/envs/lacwm-b200-py310/bin/python
WAN_DIR=$BASE/wan_fun_1.3b_control
VIDEOX_HOME=$BASE/VideoX-Fun-1d6d9c3
OFFICIAL_ABC=$BASE/src/amazon-far-abc-6bc6586

IMMUTABLE=$BASE/artifacts/dual_video_diffusion/vjepa2_cache_builds/vjepa2-cache-20260729-03-immutable1be7690
TRAIN_MANIFEST=$IMMUTABLE/manifests/train.jsonl
VAL_MANIFEST=$IMMUTABLE/manifests/val.jsonl
TRAIN_RGB_METADATA=$IMMUTABLE/caches/train/metadata.json
VAL_RGB_METADATA=$IMMUTABLE/caches/val/metadata.json
PREPROCESSED_ROOT=$BASE/data/production_v1/abc_pp
RAW_ROOT=$BASE/data/production_v1/abc_raw/data/train

RENDERER_GATE=$BASE/artifacts/dual_video_diffusion/trajectory_consistent_renderer_gate/trajectory-renderer-train384-fresh24-20260808-856cd55-v1
PARENT_RUN=$BASE/runs/dual_video_diffusion/vjepa2_controlled_study/vjepa2-faithful-cascade-20260730-seed1234-6560866-v1/vpm_parameter_matched_video
PARENT_SNAPSHOT=$PARENT_RUN/snapshot.pt
PARENT_CONFIG=$PARENT_RUN/resolved_update_1000.yaml
LEGACY_SNAPSHOT=$BASE/runs/dual_video_diffusion/vjepa2_controlled_study/vjepa2-controlled-20260730-seed1234-9cf8e69-v3/vpm_parameter_matched_video/snapshot.pt
PREFLIGHT=$BASE/artifacts/dual_video_diffusion/preflight_20260808
LINEAGE_FAILED_LOG=$PREFLIGHT/snapshot_compare-507379.log
LINEAGE_COMPARISON_LOG=$PREFLIGHT/snapshot_compare-507381.log
LPIPS_PREFLIGHT_LOG=$PREFLIGHT/lpips_pin-507388.log
CAUSAL_LADDER=$BASE/artifacts/dual_video_diffusion/causal_compressibility_ladder/causal-compressibility-train256-dev64-seed20260820-4d4db76-v1/registration.json
DIRECT_FRONTIER=$BASE/artifacts/dual_video_diffusion/vpm_direct_residual_frontier/vpm-direct-residual-fit256-outcome31-seed20260832-4f75f9c-v2

CACHE_ROOT=$BASE/artifacts/dual_video_diffusion/raw_physics_flow_cache/raw-physics-flow-cache-20260808-$SHORT-v1
STUDY_ROOT=$BASE/artifacts/dual_video_diffusion/raw_physics_flow_stage1/raw-physics-flow-stage1-20260808-$SHORT-v1
LOG_ROOT=$BASE/logs/dual_video_diffusion/raw-physics-flow-stage1-20260808-$SHORT-v1
REGISTRATION=$STUDY_ROOT/registration.json
```

The parent source is an independent, clean repository worktree at exact commit
6560866; detachment is not required by the registration contract. Both parent
source paths deliberately share the registered external Python/Wan/VideoX
runtime. Assert all immutable inputs before creating anything:

```bash
test "$(git -C "$PARENT_SOURCE_REPO" rev-parse HEAD)" = 656086686dae723c942a4209a9d71cdb17ed6ccc
test -z "$(git -C "$PARENT_SOURCE_REPO" status --porcelain --untracked-files=all)"
test "$(git -C "$OFFICIAL_ABC" rev-parse HEAD)" = 6bc6586721cf0c409ccee80f675a28de9b9b2f5e
test -z "$(git -C "$OFFICIAL_ABC" status --porcelain --untracked-files=all)"
test "$(sha256sum "$PARENT_SNAPSHOT" | awk '{print $1}')" = de65e832c56f82be1472edb1fd789e16d3a6c8a7adc9b1f31306779951cb463a
test "$(sha256sum "$PARENT_CONFIG" | awk '{print $1}')" = ae3ffd27146883917472b828c18568b72cfc7c6f2888fbca3eaa2e980a8ffd38
test "$(sha256sum "$LINEAGE_FAILED_LOG" | awk '{print $1}')" = 18bff874df2ae79bae614520ce80d3dd2d223a86d31bf1c8cc5ad24e05f10714
test "$(sha256sum "$LINEAGE_COMPARISON_LOG" | awk '{print $1}')" = 048bcddd35ecd2458e5b17f28a48a8a4967111dbb888e8cd2bc9d5ff669c0e2a
test "$(sha256sum "$LPIPS_PREFLIGHT_LOG" | awk '{print $1}')" = db37a417618afa1156cb7226140993755279191edfef8a0c628d550de20fc6af
test ! -e "$SOURCE_REPO"
test ! -e "$CACHE_ROOT"
test ! -e "$STUDY_ROOT"
test ! -e "$LOG_ROOT"
```

## 2. Freeze the audited source

Fetching is permitted only after the audit acknowledgment. The fetched branch
must resolve to the acknowledged commit before a detached worktree is made.

```bash
git -C "$SOURCE_MIRROR" fetch --no-tags origin "refs/heads/$BRANCH"
test "$(git -C "$SOURCE_MIRROR" rev-parse FETCH_HEAD)" = "$EXPECTED_COMMIT"
git -C "$SOURCE_MIRROR" worktree add --detach "$SOURCE_REPO" "$EXPECTED_COMMIT"
test "$(git -C "$SOURCE_REPO" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test -z "$(git -C "$SOURCE_REPO" status --porcelain --untracked-files=all)"
mkdir -p "$(dirname "$CACHE_ROOT")" "$(dirname "$STUDY_ROOT")" "$LOG_ROOT"
test -d "$(dirname "$CACHE_ROOT")"
test -d "$(dirname "$STUDY_ROOT")"
```

## 3. Submit the complete dependency chain

The following prefix explicitly enters Bash inside every Slurm `--wrap`
script before using `pipefail` or sourcing the B200 activation helper. Large
artifacts remain on Lustre. No protected test split is named anywhere in the
chain. The cluster's `batch` partition requires a GPU request even for the
CPU-dominant registration and sealing stages, so all four wrapped jobs request
one B200 and leave it otherwise unused where appropriate.

```bash
BASH_PREFIX="/bin/bash -lc 'set -euo pipefail; umask 077; export PYTHONDONTWRITEBYTECODE=1 LACWM_PYTHON=$PYTHON_BIN WAN_DIR=$WAN_DIR VIDEOX_HOME=$VIDEOX_HOME MUJOCO_GL=egl; source $SOURCE_REPO/tools/env/activate_b200.sh; cd $SOURCE_REPO;"

REGISTER_JOB=$(sbatch --parsable \
  --job-name=pf-register-$SHORT \
  --output="$LOG_ROOT/register-%j.out" \
  --nodes=1 --ntasks=1 --gpus-per-node=1 --cpus-per-task=32 \
  --mem=256G --time=02:00:00 \
  --partition=batch --account=coreai_chef_posttrain --qos=short --no-requeue \
  --exclude=pool0-0081,pool0-0089 \
  --wrap="$BASH_PREFIX $PYTHON_BIN tools/physics_flow_stage1.py register-cache \
    --output $CACHE_ROOT --source-repo $SOURCE_REPO \
    --expected-commit $EXPECTED_COMMIT --official-abc-root $OFFICIAL_ABC \
    --renderer-gate $RENDERER_GATE --train-manifest $TRAIN_MANIFEST \
    --val-manifest $VAL_MANIFEST --train-cache-metadata $TRAIN_RGB_METADATA \
    --val-cache-metadata $VAL_RGB_METADATA \
    --preprocessed-root $PREPROCESSED_ROOT --raw-root $RAW_ROOT'")

CACHE_TRAIN_JOB=$(sbatch --parsable --dependency=afterok:$REGISTER_JOB \
  --job-name=pf-cache-train-$SHORT \
  --output="$LOG_ROOT/cache-train-%j.out" \
  --nodes=1 --ntasks=1 --gpus-per-node=1 --cpus-per-task=32 \
  --mem=256G --time=02:00:00 --partition=batch \
  --account=coreai_chef_posttrain --qos=short --no-requeue \
  --exclude=pool0-0081,pool0-0089 \
  --wrap="$BASH_PREFIX $PYTHON_BIN tools/physics_flow_stage1.py build-cache \
    --registration $CACHE_ROOT/cache_registration.json --split train'")

CACHE_VAL_JOB=$(sbatch --parsable --dependency=afterok:$REGISTER_JOB \
  --job-name=pf-cache-val-$SHORT \
  --output="$LOG_ROOT/cache-val-%j.out" \
  --nodes=1 --ntasks=1 --gpus-per-node=1 --cpus-per-task=32 \
  --mem=256G --time=02:00:00 --partition=batch \
  --account=coreai_chef_posttrain --qos=short --no-requeue \
  --exclude=pool0-0081,pool0-0089 \
  --wrap="$BASH_PREFIX $PYTHON_BIN tools/physics_flow_stage1.py build-cache \
    --registration $CACHE_ROOT/cache_registration.json --split val'")

SEAL_JOB=$(sbatch --parsable \
  --dependency=afterok:$CACHE_TRAIN_JOB:$CACHE_VAL_JOB \
  --job-name=pf-seal-$SHORT --output="$LOG_ROOT/seal-%j.out" \
  --nodes=1 --ntasks=1 --gpus-per-node=1 --cpus-per-task=32 \
  --mem=256G --time=02:00:00 \
  --partition=batch --account=coreai_chef_posttrain --qos=short --no-requeue \
  --exclude=pool0-0081,pool0-0089 \
  --wrap="$BASH_PREFIX \
    $PYTHON_BIN tools/physics_flow_stage1.py audit-cache --metadata $CACHE_ROOT/train/metadata.json; \
    $PYTHON_BIN tools/physics_flow_stage1.py audit-cache --metadata $CACHE_ROOT/val/metadata.json; \
    $PYTHON_BIN tools/physics_flow_stage1.py register-study \
      --output $STUDY_ROOT --source-repo $SOURCE_REPO \
      --parent-source-repo $PARENT_SOURCE_REPO \
      --expected-commit $EXPECTED_COMMIT \
      --train-flow-metadata $CACHE_ROOT/train/metadata.json \
      --val-flow-metadata $CACHE_ROOT/val/metadata.json \
      --parent-snapshot $PARENT_SNAPSHOT \
      --parent-resolved-config $PARENT_CONFIG \
      --legacy-parent-snapshot $LEGACY_SNAPSHOT \
      --lineage-failed-log $LINEAGE_FAILED_LOG \
      --lineage-comparison-log $LINEAGE_COMPARISON_LOG \
      --causal-ladder-registration $CAUSAL_LADDER \
      --direct-frontier-root $DIRECT_FRONTIER \
      --lpips-preflight-log $LPIPS_PREFLIGHT_LOG \
      --python $PYTHON_BIN --wan-dir $WAN_DIR --videox-home $VIDEOX_HOME; \
    $PYTHON_BIN tools/physics_flow_stage1_workflow.py plan \
      --registration $REGISTRATION'")

FLOW_OFF_JOB=$(sbatch --parsable --dependency=afterok:$SEAL_JOB \
  --job-name=pf-off-$SHORT --output="$LOG_ROOT/train-off-%j.out" \
  "$SOURCE_REPO/tools/slurm/physics_flow_stage1.sbatch" \
  --mode train --arm FLOW-OFF --registration "$REGISTRATION" \
  --repo-root "$SOURCE_REPO" --expected-commit "$EXPECTED_COMMIT" \
  --python "$PYTHON_BIN")

RAW_FLOW_JOB=$(sbatch --parsable --dependency=afterok:$SEAL_JOB \
  --job-name=pf-raw-$SHORT --output="$LOG_ROOT/train-raw-%j.out" \
  "$SOURCE_REPO/tools/slurm/physics_flow_stage1.sbatch" \
  --mode train --arm RAW-FLOW --registration "$REGISTRATION" \
  --repo-root "$SOURCE_REPO" --expected-commit "$EXPECTED_COMMIT" \
  --python "$PYTHON_BIN")

EVAL_JOB=$(sbatch --parsable \
  --dependency=afterok:$FLOW_OFF_JOB:$RAW_FLOW_JOB \
  --job-name=pf-eval-$SHORT --output="$LOG_ROOT/eval-%j.out" \
  "$SOURCE_REPO/tools/slurm/physics_flow_stage1.sbatch" \
  --mode evaluate --registration "$REGISTRATION" \
  --repo-root "$SOURCE_REPO" --expected-commit "$EXPECTED_COMMIT" \
  --python "$PYTHON_BIN")

printf 'register=%s cache_train=%s cache_val=%s seal=%s off=%s raw=%s eval=%s\n' \
  "$REGISTER_JOB" "$CACHE_TRAIN_JOB" "$CACHE_VAL_JOB" "$SEAL_JOB" \
  "$FLOW_OFF_JOB" "$RAW_FLOW_JOB" "$EVAL_JOB"
```

## 4. Read-only monitoring and terminal evidence

Monitoring must not alter job state. The terminal success criterion is a
nonempty `$STUDY_ROOT/analysis/audit.json`, not merely a completed Slurm job.

```bash
squeue -j "$REGISTER_JOB,$CACHE_TRAIN_JOB,$CACHE_VAL_JOB,$SEAL_JOB,$FLOW_OFF_JOB,$RAW_FLOW_JOB,$EVAL_JOB" \
  -o '%.18i %.30j %.2t %.10M %.10l %R'
sacct -j "$REGISTER_JOB,$CACHE_TRAIN_JOB,$CACHE_VAL_JOB,$SEAL_JOB,$FLOW_OFF_JOB,$RAW_FLOW_JOB,$EVAL_JOB" \
  --format=JobID,JobName%32,State,ExitCode,Elapsed,Start,End
test -s "$STUDY_ROOT/analysis/audit.json"
$PYTHON_BIN "$SOURCE_REPO/tools/physics_flow_stage1.py" audit-study \
  --registration "$REGISTRATION" --read-only
```

Interpret only the sealed `analysis.json`/`audit.json`. A pass supports the
narrow causal raw-geometry scaffold claim in the prospective protocol. A stop
is evidence against this fixed-conditioning design under the registered
200-update/NFE-1 gate; it is not evidence that every possible dual video
diffusion mechanism is impossible.
