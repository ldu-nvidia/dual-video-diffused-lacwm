# Mapping Latent Forcing to LACWM

Reference implementation: `AlanBaade/LatentForcing` at
`fde8fc40377eaeeea49e6043e01c999b69779a53`.

Latent Forcing jointly corrupts RGB and frozen semantic features, embeds both
on the same token grid, sums their tokens and clock embeddings, processes them
with one transformer trunk, and predicts both states with separate heads.  Its
asymmetry comes primarily from schedule and loss masking rather than one-way
attention.

The proposed video mapping is:

| Latent Forcing | This repository |
|---|---|
| RGB pixel state | Wan video latent state |
| DINOv2 patch state | deterministic per-view TF state |
| DINO patch projection | zero-gated TF token adapter |
| shared JiT trunk | shared Wan DiT trunk |
| pixel output head | existing Wan video velocity head |
| DINO output head | new TF velocity head |
| class condition | history + planned robot actions |

## Clock conversion

The projects use opposite symbols:

```text
Latent Forcing: z = t*x + (1-t)*noise;  t=0 noise, t=1 clean
LACWM:          z = (1-sigma)*x + sigma*noise; sigma=1 noise, sigma=0 clean
```

Therefore `t = 1 - sigma`.  LACWM's velocity target is `noise - clean`, the
negative of the data-directed velocity commonly written in the other
convention.  Endpoint and Euler-sign tests are mandatory.

## Schedule families to test

1. **Aligned:** identical video and TF sigma.
2. **Independent:** clocks sampled independently; both losses active.
3. **TF leads:** both states move, but TF has lower sigma at a given progress.
4. **TF-first cascaded:** TF denoises while video stays at sigma 1, then video
   denoises while TF stays at sigma 0.
5. **TF-first cascaded-noised training:** TF-loss examples see pure-noise video;
   video-loss examples see a mostly clean TF condition with sigma in `[0,0.25]`.
6. **Video-first control:** reverse the order to test whether the mechanism, not
   simply extra parameters, creates the gain.

Both generated future states begin from noise at inference.  Clean future TF
features must never be supplied.  The final TF state is discarded after the
video latent is decoded.
