"""Batch-preprocess all downloaded ABC episodes: mcap -> per-cam mp4 + states.npz.

Parallel over episodes. Writes a manifest of successfully-processed episode dirs.
Idempotent: skips episodes whose states.npz already exists.
"""
import glob
import os
import traceback
from multiprocessing import Pool

from robot_wm.datasets.abc.preprocessing.abc_preprocess import preprocess

# Paths are env-overridable (defaults match the original /scr/ravenh layout):
#   ABC_RAW  raw episodes root, globbed as <ABC_RAW>/<task>/<episode>/episode.mcap
#   ABC_PP   output root: abc_pp/<task>/<episode>/{top,left_wrist,right_wrist}.mp4 + states.npz
ABC_RAW = os.environ.get("ABC_RAW", "/scr/ravenh/lacwm_data/abc/data/train")
ABC_PP = os.environ.get("ABC_PP", "/scr/ravenh/lacwm_data/abc_pp")
MANIFEST = os.environ.get("ABC_MANIFEST", os.path.join(ABC_PP, "manifest.txt"))
NPROC = int(os.environ.get("ABC_NPROC", "16"))


def out_dir_for(mcap_path):
    # .../abc/data/train/<task>/<episode_xxx>/episode.mcap -> abc_pp/<task>/<episode_xxx>
    rel = os.path.relpath(os.path.dirname(mcap_path), ABC_RAW)
    return os.path.join(ABC_PP, rel)


def work(mcap_path):
    out = out_dir_for(mcap_path)
    try:
        if os.path.exists(os.path.join(out, "states.npz")) and \
           os.path.exists(os.path.join(out, "top.mp4")):
            return ("skip", out)
        preprocess(mcap_path, out)
        return ("ok", out)
    except Exception as e:
        return ("err", f"{out}: {e}\n{traceback.format_exc()[:300]}")


def main():
    mcaps = sorted(glob.glob(os.path.join(ABC_RAW, "*", "*", "episode.mcap")))
    print(f"found {len(mcaps)} episodes to preprocess (nproc={NPROC})", flush=True)
    ok, skip, err = [], [], []
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
    done = ok + skip
    os.makedirs(ABC_PP, exist_ok=True)
    with open(MANIFEST, "w") as f:
        for d in sorted(done):
            f.write(d + "\n")
    print(f"DONE preprocess: ok={len(ok)} skip={len(skip)} err={len(err)} | manifest={MANIFEST} ({len(done)})", flush=True)
    for e in err[:10]:
        print("  ERR", e, flush=True)


if __name__ == "__main__":
    main()
