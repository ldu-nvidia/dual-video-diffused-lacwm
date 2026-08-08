# VideoREPA TRD screen runbook

This runbook is prospective. Do not register from a dirty checkout and do not
launch until an independent audit has accepted the committed SHA.

```bash
BASE=/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train
REPO=$BASE/src/videorepa-trd-screen/dual-video-diffused-lacwm
PY=$BASE/envs/lacwm-b200-py310/bin/python
CACHE=$BASE/artifacts/dual_video_diffusion/vjepa2_cache_builds/vjepa2-cache-20260729-03-immutable1be7690
PARENT=$BASE/runs/dual_video_diffusion/vjepa2_controlled_study/vjepa2-controlled-20260730-seed1234-9cf8e69-v3/vpm_parameter_matched_video/snapshot.pt
COMMIT=$(git -C "$REPO" rev-parse HEAD)
STUDY=$BASE/artifacts/dual_video_diffusion/videorepa_trd/videorepa-trd-seed1234-$(date +%Y%m%d)-${COMMIT:0:7}
```

Resolve the cache's registered train512 and val64 manifest/metadata filenames;
do not substitute a test manifest. Then register once:

```bash
"$PY" "$REPO/tools/videorepa_trd_screen.py" register \
  --expected-commit "$COMMIT" \
  --study-root "$STUDY" \
  --python "$PY" \
  --wan-dir "$BASE/wan_fun_1.3b_control" \
  --videox-home "$BASE/VideoX-Fun-1d6d9c3" \
  --warmstart "$PARENT" \
  --train-manifest "$CACHE/manifests/train.jsonl" \
  --train-metadata "$CACHE/caches/train/metadata.json" \
  --validation-manifest "$CACHE/manifests/val.jsonl" \
  --validation-metadata "$CACHE/caches/val/metadata.json"
```

The actual filenames in the immutable cache are authoritative; update only the
four manifest/metadata arguments if its directory layout differs. Registration
will reject wrong counts, shapes, hashes, VPM identity, source state, VideoX
commit, or W&B privacy.

Render the dependency chain without submitting:

```bash
"$REPO/tools/slurm/submit_videorepa_trd_screen.sh" \
  --registration "$STUDY/registration.json" \
  --python "$PY" \
  --expected-commit "$COMMIT" \
  --account coreai_chef_posttrain \
  --qos short
```

After reviewing the dry run and checking unrelated active jobs, launch by adding
`--execute`. The chain is:

1. exact-shape 8xB200 TRD-ON forward/backward memory canary;
2. concurrent matched TRD-OFF/TRD-ON update-200 training;
3. CPU post-training seal (no validation access);
4. concurrent target-free val64 evaluation for both arms;
5. CPU paired analysis.

Read-only monitoring:

```bash
squeue -u "$USER" -o '%.18i %.30j %.2t %.10M %.20R'
find "$STUDY" -maxdepth 3 -type f \
  \( -name 'training_complete.json' -o -name 'inventory.json' \
     -o -name 'analysis.json' \) -print
```

The final result is `$STUDY/analysis.json`. A valid result must bind the
registration and post-training seal identities, contain 576 rows per arm,
report zero teacher/target/TRD/auxiliary inference calls, cover clip indices
0--63 exactly, and state `protected_test_accessed: false`.
