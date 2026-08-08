#!/usr/bin/env bash
# Submit the prospective screen with explicit inter-job dependencies.
# Required environment variables:
#   MID_SCREEN_REGISTRATION  absolute protocol_registration.json
#   MID_SCREEN_REPO_ROOT     exact clean registered checkout
#   MID_SCREEN_PYTHON        registered B200 Python launcher

set -euo pipefail

die() {
  echo "ERROR: $*" >&2
  exit 2
}

for name in MID_SCREEN_REGISTRATION MID_SCREEN_REPO_ROOT MID_SCREEN_PYTHON; do
  [[ -n "${!name:-}" ]] || die "$name is required"
done
[[ "$MID_SCREEN_REGISTRATION" == /* && -f "$MID_SCREEN_REGISTRATION" ]] || \
  die "MID_SCREEN_REGISTRATION must be an absolute file"
REPO_ROOT="$(cd "$MID_SCREEN_REPO_ROOT" && pwd -P)"
[[ -x "$MID_SCREEN_PYTHON" ]] || die "MID_SCREEN_PYTHON is not executable"
[[ -f "$REPO_ROOT/tools/slurm/intra_forward_forcing_memory_smoke.sbatch" ]] || \
  die "memory-smoke launcher is missing"

EXPORTS="ALL,MID_SCREEN_REGISTRATION=$MID_SCREEN_REGISTRATION,MID_SCREEN_REPO_ROOT=$REPO_ROOT,MID_SCREEN_PYTHON=$MID_SCREEN_PYTHON"
LOG_ROOT="/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train/artifacts/dual_video_diffusion/intra_forward_forcing/_slurm_logs"
mkdir -p "$LOG_ROOT"

memory_raw="$(sbatch --parsable --export="$EXPORTS" \
  "$REPO_ROOT/tools/slurm/intra_forward_forcing_memory_smoke.sbatch")"
memory_job="${memory_raw%%;*}"
[[ "$memory_job" =~ ^[0-9]+$ ]] || die "invalid memory-smoke job ID: $memory_raw"

# Do not rely on Slurm array throttling for task order. MID-OFF must finish and
# publish the immutable initialization anchor before MID-ON can even start.
off_raw="$(sbatch --parsable --array=0 --dependency="afterok:$memory_job" \
  --export="$EXPORTS" \
  "$REPO_ROOT/tools/slurm/intra_forward_forcing_screen.sbatch")"
off_job="${off_raw%%;*}"
[[ "$off_job" =~ ^[0-9]+$ ]] || die "invalid MID-OFF job ID: $off_raw"

on_raw="$(sbatch --parsable --array=1 --dependency="afterok:$off_job" \
  --export="$EXPORTS" \
  "$REPO_ROOT/tools/slurm/intra_forward_forcing_screen.sbatch")"
on_job="${on_raw%%;*}"
[[ "$on_job" =~ ^[0-9]+$ ]] || die "invalid MID-ON job ID: $on_raw"

eval_raw="$(sbatch --parsable --array=0-1%2 --dependency="afterok:$on_job" \
  --export="$EXPORTS" \
  "$REPO_ROOT/tools/slurm/intra_forward_forcing_evaluate.sbatch")"
eval_job="${eval_raw%%;*}"
[[ "$eval_job" =~ ^[0-9]+$ ]] || die "invalid evaluation job ID: $eval_raw"

printf 'memory_smoke_job=%s\nmid_off_job=%s\nmid_on_job=%s\nevaluation_job=%s\n' \
  "$memory_job" "$off_job" "$on_job" "$eval_job"
