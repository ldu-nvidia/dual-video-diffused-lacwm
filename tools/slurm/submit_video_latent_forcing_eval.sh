#!/usr/bin/env bash
set -euo pipefail
umask 077

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SBATCH_SCRIPT="$REPO_ROOT/tools/slurm/video_latent_forcing_poc.sbatch"
PYTHON_BIN=""; ARM=""; DATA_ROOT=""; MANIFEST=""; CHECKPOINT=""
ARTIFACT_ROOT=""; RUN_ID=""; EXPECTED_COMMIT=""
LPIPS_LINEAR=""; LPIPS_LINEAR_SHA=""; ALEXNET=""; ALEXNET_SHA=""
R3D18=""; R3D18_SHA=""; WANDB_ENTITY_VALUE=""; WANDB_PROJECT_VALUE=""
DETERMINISM_QUALIFICATION_RECORD=""
PARTITION="batch"; ACCOUNT=""; QOS=""; CONSTRAINT="B200"
TIME_LIMIT="04:00:00"; CPUS="128"; MEMORY="900G"; EXECUTE=0
EVALUATION_SEED="20260801"

die() { echo "ERROR: $*" >&2; exit 2; }

while (($#)); do
  case "$1" in
    --python) PYTHON_BIN="$2"; shift 2 ;; --arm) ARM="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;; --manifest) MANIFEST="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;; --artifact-root) ARTIFACT_ROOT="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;; --expected-commit) EXPECTED_COMMIT="$2"; shift 2 ;;
    --seed) EVALUATION_SEED="$2"; shift 2 ;;
    --lpips-linear-weight) LPIPS_LINEAR="$2"; shift 2 ;;
    --lpips-linear-sha256) LPIPS_LINEAR_SHA="$2"; shift 2 ;;
    --alexnet-weight) ALEXNET="$2"; shift 2 ;; --alexnet-sha256) ALEXNET_SHA="$2"; shift 2 ;;
    --r3d18-weight) R3D18="$2"; shift 2 ;; --r3d18-sha256) R3D18_SHA="$2"; shift 2 ;;
    --determinism-qualification-record) DETERMINISM_QUALIFICATION_RECORD="$2"; shift 2 ;;
    --wandb-entity) WANDB_ENTITY_VALUE="$2"; shift 2 ;;
    --wandb-project) WANDB_PROJECT_VALUE="$2"; shift 2 ;;
    --partition) PARTITION="$2"; shift 2 ;; --account) ACCOUNT="$2"; shift 2 ;;
    --qos) QOS="$2"; shift 2 ;; --constraint) CONSTRAINT="$2"; shift 2 ;;
    --time) TIME_LIMIT="$2"; shift 2 ;; --cpus) CPUS="$2"; shift 2 ;;
    --mem) MEMORY="$2"; shift 2 ;;
    --execute) EXECUTE=1; shift ;;
    -h|--help) echo "usage: $0 --python PATH --arm ARM --data-root PATH --manifest PATH --checkpoint PATH --artifact-root PATH --run-id ID --expected-commit SHA --lpips-linear-weight PATH --lpips-linear-sha256 SHA --alexnet-weight PATH --alexnet-sha256 SHA --r3d18-weight PATH --r3d18-sha256 SHA [--determinism-qualification-record PATH for B0/A1/L1] [--wandb-entity E --wandb-project P] [--execute]"; exit 0 ;;
    *) die "unknown or incomplete argument: $1" ;;
  esac
done

[[ "$ARM" == phase1 || "$ARM" == B0 || "$ARM" == A1 || "$ARM" == L1 ]] || die "invalid arm"
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] || die "unsafe run ID"
[[ "$EVALUATION_SEED" == "20260801" || "$EVALUATION_SEED" == "20260802" || "$EVALUATION_SEED" == "20260803" ]] || die "invalid frozen evaluation seed"
[[ "$ARM" != phase1 || "$EVALUATION_SEED" == "20260801" ]] || die "Phase-1 evaluation seed is frozen"
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "full expected commit required"
[[ "$PYTHON_BIN" == /* ]] || die "Python path must be absolute"
PYTHON_TARGET="$(readlink -f "$PYTHON_BIN")"
[[ -f "$PYTHON_TARGET" && -x "$PYTHON_TARGET" ]] || die "invalid resolved Python"
for path in "$DATA_ROOT" "$MANIFEST" "$CHECKPOINT" "$LPIPS_LINEAR" "$ALEXNET" "$R3D18"; do
  [[ "$path" == /* ]] || die "all inputs must be absolute"
done
[[ -d "$DATA_ROOT" && ! -L "$DATA_ROOT" ]] || die "invalid data root"
for path in "$MANIFEST" "$CHECKPOINT" "$LPIPS_LINEAR" "$ALEXNET" "$R3D18"; do
  [[ -f "$path" && ! -L "$path" ]] || die "missing or symlinked input: $path"
done
if [[ "$ARM" == phase1 ]]; then
  [[ -z "$DETERMINISM_QUALIFICATION_RECORD" ]] || die "Phase-1 evaluation does not accept the Phase-2 qualification"
else
  [[ "$DETERMINISM_QUALIFICATION_RECORD" == /* && -f "$DETERMINISM_QUALIFICATION_RECORD" && ! -L "$DETERMINISM_QUALIFICATION_RECORD" ]] || \
    die "dual-arm evaluation requires an absolute, non-symlink determinism qualification"
  DETERMINISM_QUALIFICATION_RECORD="$(readlink -f "$DETERMINISM_QUALIFICATION_RECORD")"
fi
DATA_ROOT="$(readlink -f "$DATA_ROOT")"
MANIFEST="$(readlink -f "$MANIFEST")"
CHECKPOINT="$(readlink -f "$CHECKPOINT")"
LPIPS_LINEAR="$(readlink -f "$LPIPS_LINEAR")"
ALEXNET="$(readlink -f "$ALEXNET")"
R3D18="$(readlink -f "$R3D18")"
for digest in "$LPIPS_LINEAR_SHA" "$ALEXNET_SHA" "$R3D18_SHA"; do
  [[ "$digest" =~ ^[0-9a-f]{64}$ ]] || die "all weight hashes must be full lowercase SHA-256"
done
[[ "$(sha256sum "$LPIPS_LINEAR" | awk '{print $1}')" == "$LPIPS_LINEAR_SHA" ]] || die "LPIPS hash mismatch"
[[ "$(sha256sum "$ALEXNET" | awk '{print $1}')" == "$ALEXNET_SHA" ]] || die "AlexNet hash mismatch"
[[ "$(sha256sum "$R3D18" | awk '{print $1}')" == "$R3D18_SHA" ]] || die "R3D18 hash mismatch"
ARTIFACT_ROOT="$(readlink -m "$ARTIFACT_ROOT")"
[[ "$ARTIFACT_ROOT" == /* && ! -e "$ARTIFACT_ROOT/$RUN_ID" ]] || die "evaluation output must be fresh"
case "$ARTIFACT_ROOT/" in /lustre/*|/mnt/data1/*|/mnt/data2/*) ;; *) die "unapproved artifact root" ;; esac
[[ "$ARTIFACT_ROOT" != "$REPO_ROOT" && "$ARTIFACT_ROOT" != "$REPO_ROOT/"* ]] || die "artifact root cannot be inside the repository"
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || die "commit mismatch"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] || die "source worktree must be clean"
if [[ -n "$WANDB_ENTITY_VALUE" || -n "$WANDB_PROJECT_VALUE" ]]; then
  [[ -n "$WANDB_ENTITY_VALUE" && -n "$WANDB_PROJECT_VALUE" ]] || die "W&B entity/project must be paired"
fi
[[ "$CPUS" =~ ^[1-9][0-9]*$ ]] || die "--cpus must be positive"
[[ "$TIME_LIMIT" =~ ^[0-9]{2}:[0-5][0-9]:[0-5][0-9]$ ]] || die "invalid --time"

TOOL_ARGS=(eval --arm "$ARM" --data-root "$DATA_ROOT" --manifest "$MANIFEST" --checkpoint "$CHECKPOINT" --artifact-root "$ARTIFACT_ROOT" --run-id "$RUN_ID" --seed "$EVALUATION_SEED" --frontier --quality-metrics --lpips-linear-weight "$LPIPS_LINEAR" --lpips-linear-sha256 "$LPIPS_LINEAR_SHA" --alexnet-weight "$ALEXNET" --alexnet-sha256 "$ALEXNET_SHA" --r3d18-weight "$R3D18" --r3d18-sha256 "$R3D18_SHA")
[[ -n "$DETERMINISM_QUALIFICATION_RECORD" ]] && \
  TOOL_ARGS+=(--determinism-qualification-record "$DETERMINISM_QUALIFICATION_RECORD")
if [[ -n "$WANDB_ENTITY_VALUE" ]]; then
  TOOL_ARGS+=(--wandb --wandb-entity "$WANDB_ENTITY_VALUE" --wandb-project "$WANDB_PROJECT_VALUE" --wandb-private-project-ack)
fi
JOB_NAME="vlf-eval-${ARM}-${RUN_ID}"
SBATCH_ARGS=(--job-name "$JOB_NAME" --partition "$PARTITION" --constraint "$CONSTRAINT" --time "$TIME_LIMIT" --cpus-per-task "$CPUS" --mem "$MEMORY" --output "$ARTIFACT_ROOT/_slurm/%x-%j.out" --error "$ARTIFACT_ROOT/_slurm/%x-%j.err")
[[ -n "$ACCOUNT" ]] && SBATCH_ARGS+=(--account "$ACCOUNT")
[[ -n "$QOS" ]] && SBATCH_ARGS+=(--qos "$QOS")
COMMAND=(sbatch --parsable "${SBATCH_ARGS[@]}" "$SBATCH_SCRIPT" --repo-root "$REPO_ROOT" --python "$PYTHON_BIN" --expected-commit "$EXPECTED_COMMIT" -- "${TOOL_ARGS[@]}")
printf 'Resolved command (dry-run unless --execute):\n'; printf ' %q' "${COMMAND[@]}"; printf '\n'
((EXECUTE == 1)) || exit 0
command -v sbatch >/dev/null || die "sbatch unavailable"; command -v squeue >/dev/null || die "squeue unavailable"
if squeue --noheader --user "$(id -un)" --name "$JOB_NAME" | grep -q .; then die "duplicate evaluation job active"; fi
mkdir -p "$ARTIFACT_ROOT/_slurm"
JOB_ID="$("${COMMAND[@]}")"; [[ "$JOB_ID" =~ ^[0-9]+([.;][A-Za-z0-9._-]+)?$ ]] || die "unexpected sbatch response"
printf 'Submitted %s\n' "$JOB_ID"
