# TFREG screen runbook

This workflow is prospective, fresh-root only, non-requeueable, and dry-run by
default. It never accepts a protected-test path.

On the B200 login host, clone the committed branch to an immutable source path,
then set:

```bash
BASE=/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train
SRC="$BASE/src/tf-training-only-<COMMIT7>"
PY="$BASE/envs/lacwm-b200-py310/bin/python"
VIDEOX="$BASE/VideoX-Fun-1d6d9c3"
WAN="$BASE/wan_fun_1.3b_control"
CACHE="$BASE/artifacts/dual_video_diffusion/vjepa2_cache_builds/vjepa2-cache-20260729-03-immutable1be7690"
PARENT="$BASE/runs/dual_video_diffusion/vjepa2_controlled_study/vjepa2-controlled-20260730-seed1234-9cf8e69-v3/vpm_parameter_matched_video/snapshot.pt"
STUDY="$BASE/artifacts/dual_video_diffusion/tf_training_only/tfreg-seed1234-$(date -u +%Y%m%d)-<COMMIT7>-v1"
```

Use the cache's registered train512 and val64 manifest/metadata paths. Register
before any metric-bearing run:

```bash
bash "$SRC/tools/slurm/register_tf_training_only_screen.sh" \
  --python "$PY" --study-root "$STUDY" \
  --parent-snapshot "$PARENT" \
  --train-manifest "$CACHE/manifests/train.jsonl" \
  --train-metadata "$CACHE/caches/train/metadata.json" \
  --val-manifest "$CACHE/manifests/val.jsonl" \
  --val-metadata "$CACHE/caches/val/metadata.json" \
  --videox-home "$VIDEOX" --wan-dir "$WAN"
```

If the immutable cache uses different leaf names, read its inventory and pass
those exact train/val paths; do not substitute test artifacts. Registration
rehashes the parent, RGB/action arrays, manifests, and metadata and verifies the
known split identities and private personal W&B project.

Preview submission, then execute once:

```bash
COMMIT=$(git -C "$SRC" rev-parse HEAD)
bash "$SRC/tools/slurm/submit_tf_training_only_screen.sh" \
  --registration "$STUDY/registration.json" --python "$PY" \
  --expected-commit "$COMMIT"

bash "$SRC/tools/slurm/submit_tf_training_only_screen.sh" \
  --registration "$STUDY/registration.json" --python "$PY" \
  --expected-commit "$COMMIT" --execute
```

The dependency chain is one-B200 deployment canary, two matched 8xB200 training
arms, post-training seal, two target-free 8xB200 val64 evaluations, and final
analysis. Default account/QOS are `coreai_chef_posttrain/short`, wall time is two
hours, and nodes `pool0-0081,pool0-0089,pool0-0200,pool0-0343` stay excluded.

Read-only monitoring:

```bash
squeue -u "$USER" -o '%.18i %.30j %.2t %.10M %.20R'
find "$STUDY" -maxdepth 3 -type f \
  \( -name 'deployment_canary.json' -o -name 'training_complete.json' \
     -o -name 'post_training_seal.json' -o -name 'inventory.json' \
     -o -name 'analysis.json' \) -print
```

The conclusive artifact is `$STUDY/analysis.json`. Do not launch a dose follow-
up from a suggestive point estimate alone; the preregistered paired confidence
gate decides whether this mechanism earns further compute.
