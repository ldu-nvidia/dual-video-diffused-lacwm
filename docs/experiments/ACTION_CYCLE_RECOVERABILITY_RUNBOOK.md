# Action-cycle Stage-0 runbook

The submission wrapper preregisters first, runs one 8×B200 encode job for both
splits, then submits CPU ridge analysis with an `afterok` dependency. It never
opens the cached V-JEPA target or protected test.

```bash
BASE=/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train
REPO=$BASE/src/action-cycle-recoverability/dual-video-diffused-lacwm
CACHE=$BASE/artifacts/dual_video_diffusion/vjepa2_cache_builds/vjepa2-cache-20260729-03-immutable1be7690
STUDY=$BASE/artifacts/dual_video_diffusion/action_cycle_recoverability/action-cycle-stage0-seed20260809-COMMIT-v1

bash "$REPO/tools/slurm/submit_action_cycle_recoverability.sh" \
  --study-root "$STUDY" \
  --python "$BASE/envs/lacwm-b200-py310/bin/python" \
  --train-metadata "$CACHE/caches/train/metadata.json" \
  --train-manifest "$CACHE/manifests/train.jsonl" \
  --validation-metadata "$CACHE/caches/val/metadata.json" \
  --validation-manifest "$CACHE/manifests/val.jsonl" \
  --videox-home "$BASE/VideoX-Fun-1d6d9c3" \
  --wan-dir "$BASE/wan_fun_1.3b_control"
```

The wrapper intentionally sets these exact values before its first Python tool
import (ambient values are discarded):

```bash
LACWM_ALLOWED_RUN_ROOTS=$BASE
PYTHONPATH=$REPO:$BASE/VideoX-Fun-1d6d9c3
```

`BASE` is locked to the canonical, non-symlink path shown above. The generic
run-root policy remains intact: its defaults are unchanged, `/` is never
admitted, and this study is additionally restricted to a strict descendant of
`$BASE/artifacts/dual_video_diffusion/action_cycle_recoverability`. The encode
and analysis jobs independently restore and validate the same root setting.

Before submission, confirm no active duplicate study and use a fresh absent
`STUDY`. The wrapper rejects a dirty source tree, independently hashes both
large RGB/action arrays, verifies the exact clean VideoX commit and Wan asset
digests, and checks private W&B ownership. Each RGB split is fully rehashed
again immediately before encoding; train and validation actions are fully
rehashed again immediately before analysis consumes them. Do not reuse a
partial root.

Read-only monitoring:

```bash
squeue -u ldu -o '%.18i %.25j %.2t %.10M %.10l %R'
tail -n 80 "$STUDY"/logs/*.out
tail -n 80 "$STUDY"/logs/*.err
```

The conclusive artifact is `$STUDY/analysis/stage0/result.json`. A pass only
authorizes Stage 1: train a frozen clean-latent inverse-action critic, attach
its cycle loss during world-model training, remove the critic at inference,
and compare generated NFE-1 latents against shuffled-action controls plus
video-quality noninferiority. It is not itself a video-generation result.
