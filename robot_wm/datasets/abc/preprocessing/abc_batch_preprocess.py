"""Batch-preprocess all downloaded ABC episodes: mcap -> per-cam mp4 + states.npz.

Parallel over episodes. Writes a manifest of successfully-processed episode dirs.
Idempotent: skips only episodes with states.npz and all three nonempty MP4s.
"""
import glob
import os
import tempfile
import traceback
from multiprocessing import Pool

from robot_wm.datasets.abc.preprocessing.abc_preprocess import preprocess

# Paths are env-overridable (defaults match the original /scr/ravenh layout):
#   ABC_RAW  raw episodes root, globbed as <ABC_RAW>/<task>/<episode>/episode.mcap
#   ABC_PP   output root: abc_pp/<task>/<episode>/{top,left_wrist,right_wrist}.mp4 + states.npz
_LACWM_DATA = os.environ.get("LACWM_DATA", "/scr/ravenh/lacwm_data")
ABC_RAW = os.environ.get("ABC_RAW", os.path.join(_LACWM_DATA, "abc/data/train"))
ABC_PP = os.environ.get("ABC_PP", os.path.join(_LACWM_DATA, "abc_pp"))
MANIFEST = os.environ.get(
    "ABC_MANIFEST", os.path.join(ABC_PP, "manifest.success.txt")
)
INPUT_MANIFEST = os.environ.get("ABC_INPUT_MANIFEST")
NPROC = int(os.environ.get("ABC_NPROC", "16"))
REQUIRED_OUTPUTS = (
    "states.npz",
    "top.mp4",
    "left_wrist.mp4",
    "right_wrist.mp4",
)


def out_dir_for(mcap_path):
    # .../abc/data/train/<task>/<episode_xxx>/episode.mcap -> abc_pp/<task>/<episode_xxx>
    rel = os.path.relpath(os.path.dirname(mcap_path), ABC_RAW)
    return os.path.join(ABC_PP, rel)


def is_complete(out):
    """A preprocessing success must contain every file consumed by ABCDataset."""
    return all(
        os.path.isfile(os.path.join(out, name))
        and os.path.getsize(os.path.join(out, name)) > 0
        for name in REQUIRED_OUTPUTS
    )


def read_existing_successes():
    if not os.path.isfile(MANIFEST):
        return [], []
    valid, invalid = [], []
    with open(MANIFEST, "r", encoding="utf-8") as handle:
        for raw in handle:
            path = raw.strip()
            if not path:
                continue
            if is_complete(path):
                valid.append(path)
            else:
                invalid.append(f"stale/incomplete prior success: {path}")
    return valid, invalid


def write_successes_atomic(paths):
    """Merge-safe atomic write; an interrupted run leaves the prior manifest intact."""
    manifest_dir = os.path.dirname(os.path.abspath(MANIFEST))
    os.makedirs(manifest_dir, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(MANIFEST)}.", suffix=".tmp", dir=manifest_dir
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for path in sorted(set(paths)):
                handle.write(os.path.abspath(path) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, MANIFEST)
        temp_path = None
    finally:
        if temp_path is not None:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass


def work(mcap_path):
    out = out_dir_for(mcap_path)
    try:
        if is_complete(out):
            return ("skip", out)
        preprocess(mcap_path, out)
        if not is_complete(out):
            missing = [
                name
                for name in REQUIRED_OUTPUTS
                if not os.path.isfile(os.path.join(out, name))
                or os.path.getsize(os.path.join(out, name)) <= 0
            ]
            raise RuntimeError(f"preprocessing returned without required outputs: {missing}")
        return ("ok", out)
    except Exception as e:
        return ("err", f"{out}: {e}\n{traceback.format_exc()[:300]}")


def main():
    if INPUT_MANIFEST:
        with open(INPUT_MANIFEST, "r", encoding="utf-8") as handle:
            mcaps = [line.strip() for line in handle if line.strip()]
        if len(set(mcaps)) != len(mcaps):
            print("ERROR: ABC input manifest contains duplicate MCAP paths", flush=True)
            raise SystemExit(1)
        raw_root = os.path.realpath(ABC_RAW)
        invalid = [
            path
            for path in mcaps
            if not os.path.isfile(path)
            or os.path.basename(path) != "episode.mcap"
            or os.path.commonpath([raw_root, os.path.realpath(path)]) != raw_root
        ]
        if invalid:
            print(f"ERROR: invalid planned MCAPs (first 10): {invalid[:10]}", flush=True)
            raise SystemExit(1)
        mcaps = sorted(mcaps)
    else:
        mcaps = sorted(glob.glob(os.path.join(ABC_RAW, "*", "*", "episode.mcap")))
    print(f"found {len(mcaps)} episodes to preprocess (nproc={NPROC})", flush=True)
    # An explicit input manifest defines the complete active corpus; do not
    # merge stale successes from earlier download plans into it.
    prior_ok, prior_err = ([], []) if INPUT_MANIFEST else read_existing_successes()
    if not mcaps and not prior_ok:
        print("ERROR: no raw MCAPs and no prior complete successes; refusing to write an empty manifest", flush=True)
        raise SystemExit(1)
    ok, skip, err = [], [], list(prior_err)
    with Pool(NPROC) as p:
        for i, (status, info) in enumerate(p.imap_unordered(work, mcaps, chunksize=4)):
            if status == "ok":
                ok.append(info)
            elif status == "skip":
                skip.append(info)
            else:
                err.append(info)
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(mcaps)} | ok={len(ok)} skip={len(skip)} err={len(err)}", flush=True)
    # Preserve still-valid successes from earlier runs. This is important when
    # only a subset of raw MCAPs is mounted during a resumed preprocessing pass.
    done = sorted(set(prior_ok + ok + skip))
    write_successes_atomic(done)
    print(
        f"DONE preprocess: prior={len(prior_ok)} ok={len(ok)} skip={len(skip)} "
        f"err={len(err)} | success_manifest={MANIFEST} ({len(done)})",
        flush=True,
    )
    for e in err[:10]:
        print("  ERR", e, flush=True)
    if err:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
