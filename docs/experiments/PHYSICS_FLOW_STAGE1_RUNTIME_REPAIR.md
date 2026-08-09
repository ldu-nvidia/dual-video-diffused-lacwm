# Raw physics-flow Stage-1 split-runtime repairs

Date: 2026-08-09

Status: **prospective source repair; no relaunch authority**

The first fail-closed chain used exact source
`838e8da6fb69231d03cf9254a0fa79c5da27eb6c` (jobs 507546--507552) and
stopped at registration on the corrected-renderer prerequisite identity; its
source and `838e8da-v1` logs remain preserved. It is not a runtime or
scientific result.

## Preserved third failed chain: main-runtime symlink resolution

The exact audited source
`78e33861f763334b0a0d77508f2e51c6c7339770` was submitted as jobs
507620--507626. Registration and both cache builders completed; the train and
validation cache audits also passed. Seal job 507623 then failed before study
root creation because `_registered_runtime` resolved
`lacwm-b200-py310/bin/python` to the shared base CPython before launching the
offline LPIPS child. The base interpreter correctly had no LACWM
site-packages, producing `ModuleNotFoundError: No module named 'lpips'`.
`afterok` cancelled jobs 507624--507626 before either training arm or
evaluation ran.

This chain is preserved and must not be reused:

- source: `.../src/worktrees/raw-physics-flow-stage1-78e3386`;
- logs: `.../logs/dual_video_diffusion/raw-physics-flow-stage1-20260808-78e3386-v2`;
- completed cache: `.../artifacts/dual_video_diffusion/raw_physics_flow_cache/raw-physics-flow-cache-20260808-78e3386-v2`.

Read-only diagnostic allocations 507630 and 507631 are preserved in the same
log root. The first had an introspection-command quoting error. The second
recorded on compute node `pool0-0171` that the lexical entry selects
`lacwm-b200-py310` and sees the pinned LPIPS module, while the failed seal's
resolved invocation selected base CPython and lost that venv. These diagnostic
failures are provenance evidence, not scientific outcomes.

The source-only repair mirrors the cache-runtime rule. It converts the supplied
path to an absolute lexical path without resolving the final symlink, records
the complete symlink chain and resolved executable bytes, records and rehashes
`pyvenv.cfg`, requires `sys.executable` and `sys.prefix` to name that lexical
venv with user-site disabled, and content-binds its site-packages plus the
LPIPS, torch, and torchvision module and distribution `RECORD` files. Both the
runtime probe and offline LPIPS subprocess use the lexical entry. Registration
stores that same entry for all subsequent train/evaluation commands, and every
registration replay revalidates the receipt. The runbook also performs this
probe before creating a cache or study root.

Pinned main-runtime operator checks are:

| File | SHA-256 |
|---|---|
| `lacwm-b200-py310/pyvenv.cfg` | `1462a3436cb7564a778b577ed97d7b8adee292ba7f03ae92d544761a11fcbc2d` |
| `lpips-0.1.4.dist-info/RECORD` | `43d3121c0b0c2d34380a1f786dd9501be8f64270bea81f6118bbb98c384df7ae` |
| `torch-2.7.1+cu128.dist-info/RECORD` | `277c9bbc200c0507440f5b6da4b681199f7dd72250e52028e308aa81eb285d6b` |
| `torchvision-0.22.1+cu128.dist-info/RECORD` | `ae8a47757ba5c4a89c8d2f3bdbaefb7f9ca7a0cc4af350fdb58288885fde4cd2` |

A relaunch requires another independently audited source commit and fresh
commit-derived `v3` source/cache/study/log paths. Neither the successful `v2`
caches nor any failed-chain path may be resumed or overwritten.

## Preserved second failed chain: cache-renderer runtime

The exact audited source `a72d159de635d8dff3daab3a684dc4cf99c53b96`
was submitted as jobs 507569--507575. Registration completed, both cache jobs
failed before rendering because the full LACWM Python environment did not
contain MuJoCo, and all dependent jobs were cancelled. The source checkout,
logs, registration, and partial cache root remain immutable evidence and must
not be deleted or reused:

- source: `.../src/worktrees/raw-physics-flow-stage1-a72d159`;
- logs: `.../logs/dual_video_diffusion/raw-physics-flow-stage1-20260808-a72d159-v1`;
- partial cache: `.../artifacts/dual_video_diffusion/raw_physics_flow_cache/raw-physics-flow-cache-20260808-a72d159-v1`.

The failure is runtime provenance, not a model or data result. It provides no
evidence for or against raw physics-flow conditioning.

## Root cause and pinned repair

`lacwm-b200-py310/bin/python` supplies the full torch/Hydra/Wan evaluation
runtime but has no `mujoco` distribution. The existing cache-only environment
`interaction-event-py310-v1` supplies NumPy 2.0.1, MCAP 1.4.0, and MuJoCo
3.3.7. Its `bin/python` is intentionally a symlink to the LACWM venv's Python
entry, which in turn links to the shared CPython 3.10.20 binary. Resolving the
cache entry before execution loses the cache venv prefix and reproduces the
failure.

The repair therefore separates responsibilities:

1. LACWM Python performs registration, sealing/auditing, training, and
   evaluation.
2. Registration launches a child through the lexical cache-Python symlink and,
   before creating the cache root, requires exact imports/versions, an EGL
   offscreen-render preflight, and an actual decode of the deterministic first
   registered train-D405 calibration from its zstd-compressed MCAP. The child
   calibration identity must equal the full-runtime registration identity and
   the source-pinned row-1 identity
   `abc8d36380d88196e2cde46b24a57c52994fc6b518af028b89f75fffc1ee346c`.
3. The registration content-binds the lexical entry and full symlink chain,
   final CPython binary, `pyvenv.cfg`, package module files, distribution
   `RECORD` files, every RECORD-declared SHA-256/size (including native
   MuJoCo/NumPy libraries), observed unhashed entries, aggregate file
   counts/digests, helper source, and preflight receipt.
4. Only the train and validation cache builders use cache Python as their main
   process. Each reproduces the full receipt in its current process before
   creating a split directory.
5. The cache metadata, completion, cache audit, and final study registration
   carry the renderer-runtime identity transitively.

Pinned files observed before this source repair were:

| File | SHA-256 |
|---|---|
| `interaction-event-py310-v1/pyvenv.cfg` | `a85cf62de5c2c623fdc358933f86f50e58d93f41b580f097d3b1f6f66c5b67ab` |
| `mujoco-3.3.7.dist-info/RECORD` | `b403cad508902f2ea3c1106ee827cc89599e8bd3d156f5602703531ab1c1e250` |
| `numpy-2.0.1.dist-info/RECORD` | `dca51d52189d5aff4cdc2da2b4c2883c9c35ebd7e8d1cc30856b3b05c5b4bf59` |
| `mcap-1.4.0.dist-info/RECORD` | `b6c95f80a91a66103f55c58a92a0da30871e9b012beaf05a805bccdde95cdd16` |
| `mcap_protobuf_support-0.5.4.dist-info/RECORD` | `84dbce795b6b9f8ad82135443f25df5a028355805ef29668a940a31ce036d36c` |
| `protobuf-7.35.1.dist-info/RECORD` | `cb998781253fda25fd95558ea6a87879ca4897cbdcf79da2709264c8da4ce3ea` |
| `lz4-4.4.5.dist-info/RECORD` | `6ae8e7c253be6063a94f74d4478f5c3cad56e021346068851d2c7ad52ec712c0` |
| `zstandard-0.25.0.dist-info/RECORD` | `264b11507cd20241c4a087e7c2bc5f5c5a39f7ba6e589f1e509338f3fc35ce0d` |
| resolved CPython 3.10.20 binary | `49b2c58e9fddd98ff9b53f6f7613a91db9053f858c89cae5dcd25ee7828bc0d6` |

These constants are useful operator checks, but the generated registration
receipt is the authoritative full identity. A new launch requires a fresh
source commit, fresh commit-derived source/cache/study/log paths, independent
read-only source audit, and explicit authorization. It must not resume or
write into any path from jobs 507569--507575.

One pre-existing wheel metadata ambiguity is handled explicitly rather than
hidden. NumPy's RECORD contains two declarations for
`numpy/distutils/__pycache__/conv_template.cpython-310.pyc`: an unhashed row and
a stale hash/size row (declared 8,270 bytes, SHA-256
`8157884e9c3f1fff1bbe2884fdb7073b83abaa796a1c6e2e19b63022f9e597a9`).
The installed generated bytecode is 8,376 bytes with SHA-256
`2f442d2a269b70f5dfaf68e0f6355a00c27c24ccbd5f89c76654546fb87c72d9`.
The verifier permits exactly that package/version/path/two-row structure and
content-binds both declared and observed identities. It permits zero other
hash mismatches. MuJoCo has 102/102 unambiguous hash-bearing files verified,
including all 15 native libraries (19,092,432 bytes; native inventory identity
`23340861981d01decc6cd3b6e3ef14cc5ba78a918911a48d9eb8283908f7b2a6`).
This is a pre-existing generated-bytecode packaging ambiguity in the isolated
runtime, not a mutation of research source or a scientific outcome.
