#!/usr/bin/env bash
set -euo pipefail
umask 077

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SBATCH_SCRIPT="$REPO_ROOT/tools/slurm/video_latent_forcing_determinism_qualification.sbatch"
PYTHON_BIN=""; DATA_ROOT=""; MANIFEST=""; MANIFEST_SHA=""
CHECKPOINT=""; CHECKPOINT_SHA=""; R3D18=""; R3D18_SHA=""
PHASE1_GATE=""; ARTIFACT_ROOT=""; RUN_ID=""; EXPECTED_COMMIT=""
PARTITION="batch"; ACCOUNT=""; QOS=""; CONSTRAINT="B200"
TIME_LIMIT="04:00:00"; CPUS="128"; MEMORY="900G"; EXCLUDE_NODE=""; EXECUTE=0

die() { echo "ERROR: $*" >&2; exit 2; }

usage() {
  cat <<'EOF'
Submit one of the two independent deterministic-evaluation qualification jobs.
Invoke this launcher twice with distinct fresh run IDs; finalization additionally
requires that Slurm placed the jobs on disjoint nodes. Dry-run is the default.

Required: --python PATH --data-root PATH --manifest PATH --manifest-sha256 SHA
  --checkpoint PATH --checkpoint-sha256 SHA --r3d18-weight PATH
  --r3d18-sha256 SHA --phase1-gate-record PATH --artifact-root PATH
  --run-id ID --expected-commit SHA
Optional: --partition NAME --account NAME --qos NAME --constraint B200
  --time HH:MM:SS --cpus N --mem VALUE --exclude-node NODELIST --execute
EOF
}

while (($#)); do
  case "$1" in
    --python) PYTHON_BIN="$2"; shift 2 ;; --data-root) DATA_ROOT="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;; --manifest-sha256) MANIFEST_SHA="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;; --checkpoint-sha256) CHECKPOINT_SHA="$2"; shift 2 ;;
    --r3d18-weight) R3D18="$2"; shift 2 ;; --r3d18-sha256) R3D18_SHA="$2"; shift 2 ;;
    --phase1-gate-record) PHASE1_GATE="$2"; shift 2 ;;
    --artifact-root) ARTIFACT_ROOT="$2"; shift 2 ;; --run-id) RUN_ID="$2"; shift 2 ;;
    --expected-commit) EXPECTED_COMMIT="$2"; shift 2 ;;
    --partition) PARTITION="$2"; shift 2 ;; --account) ACCOUNT="$2"; shift 2 ;;
    --qos) QOS="$2"; shift 2 ;; --constraint) CONSTRAINT="$2"; shift 2 ;;
    --time) TIME_LIMIT="$2"; shift 2 ;; --cpus) CPUS="$2"; shift 2 ;;
    --mem) MEMORY="$2"; shift 2 ;; --exclude-node) EXCLUDE_NODE="$2"; shift 2 ;;
    --execute) EXECUTE=1; shift ;;
    -h|--help) usage; exit 0 ;; *) die "unknown or incomplete argument: $1" ;;
  esac
done

[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] || die "unsafe run ID"
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "full expected commit required"
[[ "$MANIFEST_SHA" =~ ^[0-9a-f]{64}$ && "$CHECKPOINT_SHA" =~ ^[0-9a-f]{64}$ && "$R3D18_SHA" =~ ^[0-9a-f]{64}$ ]] || \
  die "all input hashes must be lowercase full SHA-256"
[[ "$PYTHON_BIN" == /* && -x "$PYTHON_BIN" ]] || die "Python path must be absolute/executable"
for path in "$DATA_ROOT" "$MANIFEST" "$CHECKPOINT" "$R3D18" "$PHASE1_GATE"; do
  [[ "$path" == /* ]] || die "all input paths must be absolute"
done
[[ -d "$DATA_ROOT" && ! -L "$DATA_ROOT" ]] || die "data root is missing or symlinked"
for path in "$MANIFEST" "$CHECKPOINT" "$R3D18" "$PHASE1_GATE"; do
  [[ -f "$path" && ! -L "$path" ]] || die "input is missing or symlinked: $path"
done
DATA_ROOT="$(readlink -f "$DATA_ROOT")"; MANIFEST="$(readlink -f "$MANIFEST")"
CHECKPOINT="$(readlink -f "$CHECKPOINT")"; R3D18="$(readlink -f "$R3D18")"
PHASE1_GATE="$(readlink -f "$PHASE1_GATE")"
[[ "$(sha256sum "$MANIFEST" | awk '{print $1}')" == "$MANIFEST_SHA" ]] || die "manifest hash mismatch"
[[ "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" == "$CHECKPOINT_SHA" ]] || die "checkpoint hash mismatch"
[[ "$(sha256sum "$R3D18" | awk '{print $1}')" == "$R3D18_SHA" ]] || die "R3D-18 hash mismatch"
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || die "source commit mismatch"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" || "$EXECUTE" == 0 ]] || die "execute requires clean source"
ARTIFACT_ROOT="$(readlink -m "$ARTIFACT_ROOT")"
case "$ARTIFACT_ROOT/" in /lustre/*|/mnt/data1/*|/mnt/data2/*) ;; *) die "unapproved artifact root" ;; esac
[[ "$ARTIFACT_ROOT" != "$REPO_ROOT" && "$ARTIFACT_ROOT" != "$REPO_ROOT/"* ]] || die "artifact root cannot be in Git"
[[ ! -e "$ARTIFACT_ROOT/$RUN_ID" ]] || die "qualification output must be fresh"
[[ "$CONSTRAINT" == "B200" ]] || die "qualification constraint is frozen to exact B200"
[[ "$CPUS" =~ ^[1-9][0-9]*$ && "$TIME_LIMIT" =~ ^[0-9]{2}:[0-5][0-9]:[0-5][0-9]$ ]] || die "invalid resources"
if [[ -n "$EXCLUDE_NODE" ]]; then
  [[ "$EXCLUDE_NODE" =~ ^[A-Za-z0-9][A-Za-z0-9_.\[\],-]*$ ]] || \
    die "unsafe --exclude-node value"
fi

TOOL_ARGS=(run --data-root "$DATA_ROOT" --manifest "$MANIFEST" --manifest-sha256 "$MANIFEST_SHA"
  --checkpoint "$CHECKPOINT" --checkpoint-sha256 "$CHECKPOINT_SHA"
  --r3d18-weight "$R3D18" --r3d18-sha256 "$R3D18_SHA"
  --phase1-gate-record "$PHASE1_GATE" --artifact-root "$ARTIFACT_ROOT" --run-id "$RUN_ID"
  --expected-source-commit "$EXPECTED_COMMIT" --expected-python "$PYTHON_BIN")
JOB_NAME="vlf-detqual-${RUN_ID}"
SBATCH_ARGS=(--job-name "$JOB_NAME" --partition "$PARTITION" --constraint "$CONSTRAINT"
  --time "$TIME_LIMIT" --cpus-per-task "$CPUS" --mem "$MEMORY"
  --output "$ARTIFACT_ROOT/_slurm/%x-%j.out" --error "$ARTIFACT_ROOT/_slurm/%x-%j.err")
[[ -n "$ACCOUNT" ]] && SBATCH_ARGS+=(--account "$ACCOUNT")
[[ -n "$QOS" ]] && SBATCH_ARGS+=(--qos "$QOS")
[[ -n "$EXCLUDE_NODE" ]] && SBATCH_ARGS+=(--exclude "$EXCLUDE_NODE")
COMMAND=(sbatch --parsable "${SBATCH_ARGS[@]}" "$SBATCH_SCRIPT" --repo-root "$REPO_ROOT"
  --python "$PYTHON_BIN" --expected-commit "$EXPECTED_COMMIT" -- "${TOOL_ARGS[@]}")
printf 'Resolved command (dry-run unless --execute):\n'; printf ' %q' "${COMMAND[@]}"; printf '\n'
((EXECUTE == 1)) || exit 0
command -v sbatch >/dev/null || die "sbatch unavailable"; command -v squeue >/dev/null || die "squeue unavailable"
if squeue --noheader --user "$(id -un)" --name "$JOB_NAME" | grep -q .; then die "duplicate qualification job active"; fi
mkdir -p "$ARTIFACT_ROOT/_slurm"
JOB_ID="$("${COMMAND[@]}")"; [[ "$JOB_ID" =~ ^[0-9]+([.;][A-Za-z0-9._-]+)?$ ]] || die "unexpected sbatch response"
printf 'Submitted %s\n' "$JOB_ID"
