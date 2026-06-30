# B200 runtime

This directory makes the Wan dependency reproducible without modifying the
upstream checkout or importing all of VideoX-Fun.

## Pinned source

VideoX-Fun is pinned to commit
`1d6d9c3e1540968466937129fef4b288041e06de` (2025-09-12). This is the earliest
audited upstream revision that both preserves the Wan2.1 APIs used here and
uses the post-Diffusers-0.33 `model_loading_utils` path needed by Diffusers
0.38.0. Earlier revision `9c16b3a73f7739e643a4d4b89fbd9c88e8477fb4`
has a smaller package initializer, but its requested low-memory load falls back
to full model construction under Diffusers 0.38.0.

The `videox_shim` namespace overlay prevents eager imports of unrelated
VideoX-Fun image/audio models and xFuser backends. The actual Wan transformer,
VAE, attention, and config code still comes from the pinned upstream checkout.
The shim intentionally disables VideoX-Fun sequence-parallel *inference*;
lacwm training uses its own PyTorch DDP path.

## Create and verify

```bash
tools/env/create_b200_env.sh
source /mnt/data2/$USER/lacwm_runtime/envs/lacwm-b200-py310/bin/activate
source tools/env/activate_b200.sh
tools/env/prepare_wan_assets.sh --device cuda:0
python tools/env/verify_b200_runtime.py
python tools/env/verify_b200_runtime.py --expected-gpus 8 --require-b200
```

If Python 3.10.20 is absent, the bootstrap checksum-verifies pinned `uv` 0.10.9
and installs both the tool and managed Python under `LACWM_BASE`; it does not
modify shell startup files. All caches follow the same data-volume root.

The CUDA 12.8 PyTorch wheel contains the runtime libraries; a system CUDA
toolkit is not required unless compiling additional extensions. The NVIDIA
driver must support the installed B200s and CUDA 12.8. Do not install optional
FlashAttention, SageAttention, or xFuser packages until the baseline DDP smoke
test passes; the audited path uses PyTorch SDPA.

`prepare_wan_assets.sh` pins Hugging Face revision
`ce96ebd52b1134d2c8a903ceb491ab27aa1e5b7c`, downloads only the model files
needed by training, and reproducibly creates the cached empty-prompt embedding.
The explicit device is used only while building that cache; an existing valid
cache is reused without loading the text encoder.

`verify_b200_runtime.py` checks Python 3.10.20 and the complete 126-distribution
inventory against the resolved lock, plus the clean upstream Git commit, Wan
loader/forward signatures, checkpoint sizes and SHA-256 digests, null-prompt
shape/finiteness/provenance hash, GPU count/capability, BF16, NCCL, and peer access. It does not
instantiate the 1.7B model. Checkpoint validation is opt-in with
`--wan-dir "$WAN_DIR"`.

The production launcher does not use Hydra Submitit. `debugpy` is optional and
only needed with `debug=true`; install it separately in a development runtime.

`requirements-b200.txt` documents the direct dependency contract;
`requirements-b200-lock.txt` is the fully resolved environment installed by the
bootstrap script.
