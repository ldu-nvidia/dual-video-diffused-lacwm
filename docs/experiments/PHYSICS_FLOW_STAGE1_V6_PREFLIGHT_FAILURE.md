# Raw physics-flow Stage-1 `b24d13a-v6` preflight failure

Date: 2026-08-09

Status: **registration failed before cache/study creation; zero training and
zero evaluation; no scientific result**

The strict v6 chain was launched from clean source commit
`b24d13ac5d463d328ff6f7251d614154f466e9af`. Registration job `507810`
failed after the compute/runtime preflight and before `register-cache` created
its output root. Jobs `507811`--`507816` were dependency-cancelled, so no cache,
study, arm, endpoint, target, metric, or W&B outcome was produced.

| Job | Stage | Terminal state | Exit | Elapsed |
|---:|---|---|---:|---:|
| 507810 | registration/preflight | `FAILED` | `2:0` | `00:00:54` |
| 507811 | train cache | `CANCELLED` | `0:0` | `00:00:00` |
| 507812 | validation cache | `CANCELLED` | `0:0` | `00:00:00` |
| 507813 | audit/study seal | `CANCELLED` | `0:0` | `00:00:00` |
| 507814 | `FLOW-OFF` training | `CANCELLED` | `0:0` | `00:00:00` |
| 507815 | `RAW-FLOW` training | `CANCELLED` | `0:0` | `00:00:00` |
| 507816 | evaluation | `CANCELLED` | `0:0` | `00:00:00` |

The failure was:

```text
joint_states.npy has unsupported local ZIP metadata
```

The exact registration log is 11,374 bytes with SHA-256
`5ee317f67dc8b1b301228b9742546de7e5f1e5e84aa76850fec840f595f549e7` at:

```text
/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/
  lacwm_train/logs/dual_video_diffusion/
  raw-physics-flow-stage1-20260808-b24d13a-v6/register-507810.out
```

Both intended output paths were confirmed absent after terminal job state.

## Root cause and bounded repair

The parser accepted current NumPy's streamed ZIP64 local layout (version 45,
32-bit size sentinels, one 20-byte ZIP64 size field), plus plain 32-bit ZIP
members. The immutable ABC archives were written by an older NumPy behavior:
each local header uses version 20 and valid 32-bit sizes while redundantly
carrying the same 20-byte ZIP64 size field. The central directory uses version
20, ordinary sizes, and no extra field. This remains a `ZIP_STORED`, directly
addressable layout; the extra field is metadata, not array data.

A read-only structural scan of all registered train/validation manifests
examined local headers, end-of-central-directory, and all central records but
zero member payload bytes. It found exactly 576 archives and 3,456 members;
all 3,456 used the same version-20 redundant-ZIP64 class. Every redundant size
matched both local 32-bit sizes, compression was `ZIP_STORED`, flags were zero,
member order/count matched, and central/local metadata agreed.

The v7 repair admits only that exact 20-byte ZIP64 field and requires both
redundant sizes to equal the local-header sizes. Arbitrary extras, inconsistent
sizes, compression, flags, unsupported versions, or central/local differences
still fail before any NPY payload read. Tests cover successful selected-byte
access and a corrupted redundant-size failure with metadata-only reads.

This is an engineering preflight failure, not evidence for or against raw
physics-flow conditioning. Only a fresh v7 namespace may yield a quality
decision.
