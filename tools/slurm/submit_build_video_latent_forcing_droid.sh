#!/usr/bin/env bash
set -euo pipefail
umask 077

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SBATCH_SCRIPT="$REPO_ROOT/tools/slurm/build_video_latent_forcing_droid.sbatch"
PYTHON_BIN=""
EXPECTED_COMMIT=""
DATA_ROOT=""
OUTPUT_ROOT=""
PARTITION="batch"
ACCOUNT=""
QOS=""
CONSTRAINT="b200"
TIME_LIMIT="04:00:00"
CPUS="128"
MEMORY="900G"
EXECUTE=0

die() { echo "ERROR: $*" >&2; exit 2; }

while (($#)); do
  case "$1" in
    --python) [[ $# -ge 2 ]] || die "$1 requires a value"; PYTHON_BIN="$2"; shift 2 ;;
    --expected-commit) [[ $# -ge 2 ]] || die "$1 requires a value"; EXPECTED_COMMIT="$2"; shift 2 ;;
    --data-root) [[ $# -ge 2 ]] || die "$1 requires a value"; DATA_ROOT="$2"; shift 2 ;;
    --output-root) [[ $# -ge 2 ]] || die "$1 requires a value"; OUTPUT_ROOT="$2"; shift 2 ;;
    --partition) [[ $# -ge 2 ]] || die "$1 requires a value"; PARTITION="$2"; shift 2 ;;
    --account) [[ $# -ge 2 ]] || die "$1 requires a value"; ACCOUNT="$2"; shift 2 ;;
    --qos) [[ $# -ge 2 ]] || die "$1 requires a value"; QOS="$2"; shift 2 ;;
    --constraint) [[ $# -ge 2 ]] || die "$1 requires a value"; CONSTRAINT="$2"; shift 2 ;;
    --time) [[ $# -ge 2 ]] || die "$1 requires a value"; TIME_LIMIT="$2"; shift 2 ;;
    --cpus) [[ $# -ge 2 ]] || die "$1 requires a value"; CPUS="$2"; shift 2 ;;
    --mem) [[ $# -ge 2 ]] || die "$1 requires a value"; MEMORY="$2"; shift 2 ;;
    --execute) EXECUTE=1; shift ;;
    -h|--help)
      echo "usage: $0 --python PATH --expected-commit SHA --data-root PATH --output-root PATH [--partition P --account A --qos Q --constraint C --time HH:MM:SS --cpus N --mem M] [--execute]"
      exit 0
      ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "$PYTHON_BIN" == /* ]] || die "Python path must be absolute"
PYTHON_TARGET="$(readlink -f "$PYTHON_BIN")"
[[ -f "$PYTHON_TARGET" && -x "$PYTHON_TARGET" ]] || die "invalid resolved Python"
[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "full --expected-commit is required"
[[ -d "$DATA_ROOT" && ! -L "$DATA_ROOT" ]] || die "invalid DROID source root"
DATA_ROOT="$(readlink -f "$DATA_ROOT")"
OUTPUT_ROOT="$(readlink -m "$OUTPUT_ROOT")"
[[ "$OUTPUT_ROOT" == /* && ! -e "$OUTPUT_ROOT" ]] || die "output must be a fresh absolute path"
case "$OUTPUT_ROOT/" in /lustre/*|/mnt/data1/*|/mnt/data2/*) ;; *) die "unapproved output root" ;; esac
[[ "$OUTPUT_ROOT" != "$REPO_ROOT" && "$OUTPUT_ROOT" != "$REPO_ROOT/"* ]] || die "output cannot be in repo"
[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || die "commit mismatch"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] || die "source worktree must be clean"
[[ "$CPUS" =~ ^[1-9][0-9]*$ ]] || die "--cpus must be positive"
[[ "$TIME_LIMIT" =~ ^[0-9]{2}:[0-5][0-9]:[0-5][0-9]$ ]] || die "invalid --time"

SBATCH_ARGS=(--job-name vlf-droid-cache --partition "$PARTITION" --constraint "$CONSTRAINT" --time "$TIME_LIMIT" --cpus-per-task "$CPUS" --mem "$MEMORY" --output "$(dirname "$OUTPUT_ROOT")/_slurm/vlf-droid-cache-%j.out" --error "$(dirname "$OUTPUT_ROOT")/_slurm/vlf-droid-cache-%j.err")
[[ -n "$ACCOUNT" ]] && SBATCH_ARGS+=(--account "$ACCOUNT")
[[ -n "$QOS" ]] && SBATCH_ARGS+=(--qos "$QOS")
COMMAND=(sbatch --parsable "${SBATCH_ARGS[@]}" "$SBATCH_SCRIPT" --python "$PYTHON_BIN" --expected-commit "$EXPECTED_COMMIT" --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT")
printf 'Resolved command (dry-run unless --execute):\n'
printf ' %q' "${COMMAND[@]}"
printf '\n'
((EXECUTE == 1)) || exit 0
command -v sbatch >/dev/null || die "sbatch unavailable"
command -v squeue >/dev/null || die "squeue unavailable"
if squeue --noheader --user "$(id -un)" --name vlf-droid-cache | grep -q .; then
  die "a cache build with this job name is already active"
fi
mkdir -p "$(dirname "$OUTPUT_ROOT")/_slurm"
JOB_ID="$("${COMMAND[@]}")"
[[ "$JOB_ID" =~ ^[0-9]+([.;][A-Za-z0-9._-]+)?$ ]] || die "unexpected sbatch response: $JOB_ID"
printf 'Submitted %s\n' "$JOB_ID"
