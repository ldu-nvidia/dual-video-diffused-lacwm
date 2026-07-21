# Production LACWM provenance

The bootstrap base is the completed LoRA production lineage on GCP-NRT:

| Field | Value |
|---|---|
| Source commit | `f227b3b5108cd63c0fc853a08a26ca705606387c` |
| Source branch | `codex/topology-migration-8n` |
| Run | `lora8n-stage2-fastall4-v1-f227b3b-posttrain-mig1` |
| Completion | 60,000 / 60,000 optimizer updates |
| Completion time | `2026-07-13T23:52:04.636659+00:00` |
| Nodes / GPUs | 8 nodes / 64 B200 GPUs |
| Effective global batch | 1,024 |
| Variant | latent action, rank-64 LoRA |
| Runtime | Python 3.10.20, PyTorch 2.7.1+cu128 |
| VideoX-Fun | `1d6d9c3e1540968466937129fef4b288041e06de` |
| Base model | Wan2.1-Fun-1.3B-Control |
| Run identity SHA-256 | `a35827f11a9dadb7e0a4aca39d43a62637feb12d71f557bf86009717eb95b98d` |

The reference corpus contained capped DROID, EgoDex, AgiBot, and ABC inputs.
The run identity records expected episode counts of 10,000, 10,000, 5,671, and
10,000 respectively, together with manifest and filesystem fingerprints.

Cluster locations are recorded for reproducibility, not embedded as runtime
defaults in new code:

```text
project root:
/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train

source:
.../src/lacwm_dit_lora_B200_8n_topology_migration_f227b3b

checkpoint:
.../runs/production/lora8n-stage2-fastall4-v1-f227b3b-posttrain-mig1/snapshot.pt
```

## Baseline architecture

```text
13 RGB frames = 5 history + 8 future
3 camera views width-stacked at 180x960
        |
frozen causal Wan VAE (8x spatial, 4x temporal)
        v
video latent [B,16,4,24,120]

Wan input channels:
  noisy video 16 + action control 16 + history reference 16 = 48

Wan DiT:
  30 blocks, width 1536, 12 heads, output 16-channel video velocity
```

The initial dual-diffusion work must load the production checkpoint with the
new feature disabled and reproduce its fixed-seed output before any training.
No checkpoint or dataset is tracked in Git.
