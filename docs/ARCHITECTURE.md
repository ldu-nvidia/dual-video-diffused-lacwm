# Target architecture and tensor contracts

## Data path

```text
raw video [B,13,3,180,960]
  |                                   |
  | frozen Wan VAE                    | split 3 views before any transform
  v                                   v
video [B,16,4,24,120]       pad each view 180x320 -> 192x320,
                            then area-downsample -> 24x40
                                      |
                                      v
                           causal local RFFT or STFT
                                      |
                                      v
                              TF [B,Ctf,4,24,120]
```

The recommended first target is the complete length-4 real FFT aligned to Wan's
causal bins.  For RGB, each real four-sample window becomes 12 channels:

```text
[X0.real, X1.real, X1.imag, X2.real] x RGB
```

Unlike magnitude-only features, this representation retains phase and is
exactly invertible at its downsampled spatial scale.  The 13-frame mapping is:

```text
bin 0: frame 0 repeated as the causal anchor
bin 1: frames 1--4
bin 2: frames 5--8
bin 3: frames 9--12
```

Localized Hann-window STFT, wavelets, magnitude-only coefficients, high-pass
coefficients, and transforms over per-frame VAE features remain ablations.

## Dual flow

```text
zv(sv)  = (1-sv)  * video_clean + sv  * epsilon_v
ztf(stf)= (1-stf) * tf_clean    + stf * epsilon_tf

target_v  = epsilon_v  - video_clean
target_tf = epsilon_tf - tf_clean
```

The states have independent noise, clocks, loss masks and sampler increments.
History and invalid camera regions require explicit masks for both losses.

## Wan integration boundary

Do not widen Wan's pretrained 48-channel patch input.  The intended integration
is additive at the token level:

```text
pretrained Wan patch tokens(noisy video + action + history)
       + tanh(gate=0) * TFAdapter(noisy TF)
       + TF clock embedding
                         |
                         v
                 shared 30-block Wan trunk
                    /                \
existing video velocity head     zero-init TF velocity head
```

With the gate and TF head initialized to zero, `dual.enabled=false` and the
initial `dual.enabled=true` forward path must preserve the original video output
exactly before optimization.  The adapter uses the existing `(1,2,2)` Wan patch
grid, so it does not increase attention sequence length.

The production VideoX-Fun transformer currently returns only its final video
output.  Exposing shared hidden tokens and inserting the TF residual is a Phase
3 change; it is deliberately not monkey-patched in the bootstrap commit.

## Causal deployment contract

The deployed sampler may consume only:

- observed history RGB;
- proposed future robot actions;
- requested output shape and sampling noise.

It may not encode ground-truth future RGB to infer actions, construct TF
conditioning, obtain validity masks, or determine latent shape.  The
explicit-action variant is therefore the causal starting point, even though the
completed production reference checkpoint uses the latent-action variant.
