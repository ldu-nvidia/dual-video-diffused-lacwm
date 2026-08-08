# ABC D405 nominal geometry probe

## Question

Can the official YAM MuJoCo model provide an inference-valid, action-derived
robot-motion scaffold for ABC video generation, without using clean future
video features?

This first probe tests only the prerequisite: whether a nominal robot render is
better aligned with observed train RGB when driven by the matching observed
pose than when driven by a time-shifted pose. It is not a generator-quality
experiment.

## Leakage contract

- Input rows must carry `split=train`; validation and test rows are rejected.
- Every bundle records `protected_test_accessed=false` and is hash checked.
- No future-video feature is supplied to a generator.
- Observed states are used only for this geometry/calibration diagnostic. A
  deployable scaffold must replay the action trajectory available before RGB
  denoising.

## Pinned geometry and mapping

- Official source: `amazon-far/abc` commit
  `6bc6586721cf0c409ccee80f675a28de9b9b2f5e`.
- Official scene: `assets/put_bottles/put_bottle.xml`.
- Only the visual YAM meshes, two arms, and nominal D405 cameras are retained;
  task objects and collision geometry are excluded from rendering.
- LACWM ABC order is `[left arm 6, right arm 6, left grip, right grip]`.
- Official ABC sim order is `[left arm 6, left grip, right arm 6, right grip]`.
  The exact cache permutation is `[0,1,2,3,4,5,12,6,7,8,9,10,11,13]`.
- Gripper aperture is mapped from `[0,1]` to paired finger qpos
  `+/- aperture * 0.0475`.

## Probe

For each selected D405 train clip:

1. extract the registered 13 top-camera frames, matching observed joint and
   gripper states, and raw MCAP intrinsics;
2. derive MuJoCo vertical FOV from the episode's `fy` while retaining the
   official nominal camera extrinsic;
3. render a robot-only segmentation mask for the aligned pose;
4. render a cyclic four-frame time-shifted pose as the negative control;
5. compare each rendered silhouette boundary with Canny edges in observed RGB
   using mean edge distance and support within three pixels;
6. bootstrap the paired shifted-minus-aligned distance across frames;
7. save aligned/control overlays and render latency.

The exploratory diagnostic gate requires a positive 95% bootstrap lower bound
for shifted-minus-aligned edge distance and higher aligned three-pixel support.
Even a pass establishes only nominal geometric signal; it does not establish
video-generation benefit.

## Usage

```bash
python tools/abc_d405_nominal_geometry_probe.py extract \
  --clip-manifest /path/to/train.jsonl \
  --preprocessed-root /path/to/abc_pp \
  --raw-root /path/to/abc_raw/data/train \
  --clip-id CLIP_ID --max-clips 3 \
  --output-dir /path/to/bundles

MUJOCO_GL=egl python tools/abc_d405_nominal_geometry_probe.py evaluate \
  --bundle-dir /path/to/bundles \
  --official-abc-root /path/to/amazon-far-abc \
  --output-dir /path/to/results
```

Results and interpretation are appended only after the train-only run is
complete.
