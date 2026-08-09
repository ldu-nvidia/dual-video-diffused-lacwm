# Physics-flow Stage-1 parent-lineage decision

Date audited: 2026-08-08

This is a pre-outcome checkpoint audit. No cache rendering, video continuation
training, causal endpoint sampling, or future validation RGB access occurred.

## Canonical comparison

The original scaffold named this older VPM snapshot:

- path: `/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train/runs/dual_video_diffusion/vjepa2_controlled_study/vjepa2-controlled-20260730-seed1234-9cf8e69-v3/vpm_parameter_matched_video/snapshot.pt`
- file SHA-256: `f67c7bae50c4c279bf6372e098833be32699aca24232d7d489a1f7a45b5a8e21`
- run identity: `649a2c11a0a77091ed6e8d54073dd45a825239dfe3b0245ca5a55876c4df9fba`
- canonical model-state SHA-256: `2b29819672da71cde9d8732e619a50fc2ab941996620b0de8a9d93edcd1a1d9c`

The faithful-cascade causal-compressibility frontier instead used:

- path: `/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train/runs/dual_video_diffusion/vjepa2_controlled_study/vjepa2-faithful-cascade-20260730-seed1234-6560866-v1/vpm_parameter_matched_video/snapshot.pt`
- file SHA-256: `de65e832c56f82be1472edb1fd789e16d3a6c8a7adc9b1f31306779951cb463a`
- run identity: `d79c3699f0c68dcd17321fcb0ea2846fca4d96f8c815f91dff02a19d3140787f`
- canonical model-state SHA-256: `d1231b8bc13a2391a94f2ade8ff216de3fbe5e91e7242b35c39c60197fd897a0`

Both checkpoints contain 1,686 tensors and have the same canonical schema
SHA-256, `9626c924be5ead5fdf5dc7d5887e7435a621845b1b3f739cd72344d94404b97a`,
but 495 tensors are byte-distinct:

| Family | Different tensors |
|---|---:|
| Wan self/cross-attention LoRA A/B | 480 |
| action encoder | 6 |
| action pool | 4 |
| action-to-control | 4 |
| morphology embedding | 1 |

Therefore `f67…` is a distinct older model, not a container-only variant of
the actual frontier. It is rejected as the Stage-1 parent. Both matched arms
must start from `de65…/d79c…`; `FLOW-OFF` remains the paired baseline, and
`RAW-FLOW` must add useful causal structure beyond this same frontier's
ordinary direct-residual capacity.

## Immutable evidence

- The first read-only comparison attempt, Slurm job `507379`, failed before a
  result because the harness did not flatten a 0-D tensor before byte view.
  Its preserved log is 849 bytes with SHA-256
  `18bff874df2ae79bae614520ce80d3dd2d223a86d31bf1c8cc5ad24e05f10714`.
- The corrected read-only comparison, job `507381`, completed successfully.
  Its full 495-name inventory is 37,385 bytes with SHA-256
  `048bcddd35ecd2458e5b17f28a48a8a4967111dbb888e8cd2bc9d5ff669c0e2a`.
- Both logs live under
  `/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train/artifacts/dual_video_diffusion/preflight_20260808/`.
- The causal-compressibility ladder registration is
  `/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train/artifacts/dual_video_diffusion/causal_compressibility_ladder/causal-compressibility-train256-dev64-seed20260820-4d4db76-v1/registration.json`,
  identity `3517eb729ee1b47576ca6a91159121f24fda2b73b7127dcb468628807356b36a`.
  It binds VPM arm identity
  `d79c3699f0c68dcd17321fcb0ea2846fca4d96f8c815f91dff02a19d3140787f`,
  stage identity
  `17149ca2619afed49d933f03c3613c56d86cb858f5a7fe1402a76e1ff9266d00`,
  and outcome identity
  `249b9e755f4aeefcd24e52169cb9f03a3f53815f860a35912c32098c0c741172`.
- The direct-residual frontier root is
  `/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train/artifacts/dual_video_diffusion/vpm_direct_residual_frontier/vpm-direct-residual-fit256-outcome31-seed20260832-4f75f9c-v2/`.
  Registration identity is `e9d4ce9c6299c42d4da0f92302cd8d9ba03ca0b3fcaf6b5c8e622e9707621448`;
  fit `4bfb8a6085db657bbf846f03d3880d4c115e2a4ca025a9395bd75a62e9e1fe7d`;
  endpoint `6d3d4f4f1afeef5bb85046655d33bf019f1fb5367028edbee9d9e9468034242c`;
  analysis `6b15b3865925745f2e889585078f66238707315d695ddebad63ae43307ecf67d`;
  completion `0bf783dccb6105cbd7166014e71bfa24d77d3a91aff624b6e8602532ba16dffc`;
  and audit `db093d20647829b67e515f72f4b56f240ddf935a8c11973d2bc917d20b3140fb`.
  Its frozen source commit is `4f75f9c08dbd64f7ad7d13373a98f25e397f7f4b`.

The Stage-1 prospective registration must content-bind both logs, both
snapshots, this note, the ladder lineage, and the direct-residual frontier
lineage before any new outcome is opened.
