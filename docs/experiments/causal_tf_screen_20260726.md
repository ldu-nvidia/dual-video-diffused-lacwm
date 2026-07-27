# Causal TF-conditioning screen (pre-data specification)

Date: 2026-07-26

Dataset: ABC only

Model: explicit-action LACWM, Wan latent grid

Clock convention: `sigma=1` is noise; `sigma=0` is clean data.

## Question

Does aligned causal-RFFT content help the video branch learn or sample better,
as opposed to any benefit caused by an extra residual, parameters, or
regularization?

The completed 100-update pilot cannot answer this question because the learned
TF-state gate in the conditioned arm ended at approximately `-3.75e-4`. It also
compared that token RMS to raw TF coefficients rather than to Wan tokens at the
fusion seam.

## Intervention

The auxiliary TF encoder/head now receives its own full normalized TF tokens.
A separate state gate controls only the residual injected into the video
trunk. The gate is fixed during the screen so the model cannot evade the test
by closing it.

All arms start from the reviewed production model checkpoint, use seed 1234,
train for 200 optimizer updates, and differ only in TF content exposed to the
video trunk and its fixed scale:

| Arm | Video-visible future TF | Fixed state scale |
|---|---|---:|
| `off_s000` | none | 0.00 |
| `matched_s003` | same sample | 0.03 |
| `shuffled_s003` | another sample | 0.03 |
| `matched_s010` | same sample | 0.10 |
| `shuffled_s010` | another sample | 0.10 |

For the shuffled control, the observed TF history stays correct. The hidden
future clean TF is deranged across the global batch, then corrupted using the
local sample's sigma and noise. Thus matched versus shuffled retains the same
history, clock, noise marginal, architecture, parameter count, and residual
scale.

## Exposure qualification

Training must log:

- normalized TF-state token RMS;
- TF-state residual RMS;
- clock residual RMS;
- native Wan patch-embedding RMS before TF addition;
- state-residual/native-token RMS ratio;
- combined-residual/native-token RMS ratio.

A scale is not interpretable if the measured state ratio is negligible,
non-finite, or unstable. The `0.10` arm is a stress point; failure at this
scale does not invalidate a stable `0.03` result.

## Inference tests

Each visualization rank uses a fixed evaluation seed and stores the exact
initial video and TF states. The following are independent sampler runs—not
prefixes of an eight-step trajectory:

- NFE: 1, 2, 4, and 8;
- autonomous TF;
- conditioning disabled;
- correctly aligned scheduled ground-truth TF (oracle);
- wrong-sample scheduled ground-truth TF (oracle control).

Oracle inputs leak the hidden future and are diagnostic upper bounds, never
deployable results. At one NFE both future states begin at pure noise, so no
aligned autonomous-TF benefit is structurally possible; one NFE is a negative
control. A deployable benefit can first arise on call two.

## Metrics and comparisons

Primary screen endpoints:

1. future video-latent NMSE at 4 NFE;
2. decoded temporal-difference MSE at 4 NFE.

Supporting endpoints:

- decoded MSE and PSNR at 1/2/4/8 NFE;
- future TF-latent NMSE;
- the same video endpoints at 8 NFE;
- paired per-rank differences and fixed-seed bootstrap 95% intervals.

FVD is not a primary endpoint for the small rank-level probe. The existing
temporally averaged Inception implementation is not canonical I3D FVD.

## Decision rules

The current TF integration is a promising causal signal only if matched beats
shuffled at the same scale by at least 3% on the primary 4-NFE temporal metric,
does not regress video-latent NMSE, and the direction agrees at 8 NFE. It then
advances to three fresh seeds from the original production checkpoint.

Interpret oracle interventions as follows:

- correct oracle better than wrong oracle, autonomous null: the representation
  is usable, but the TF denoiser/schedule is the bottleneck;
- correct oracle indistinguishable from wrong oracle at both verified exposure
  levels: this causal-RFFT representation/injection path is likely unhelpful;
- autonomous matched better than autonomous shuffled/off: deployable TF
  conditioning is working.

A scoped negative conclusion requires verified exposure through at least the
0.10 scale and matched-versus-shuffled effects inside a +/-2% equivalence band,
including the oracle comparison. It applies only to ABC, this complete
length-four causal RFFT, this Wan fusion seam, and the tested schedules. It
does not establish that all time-frequency video representations are useless.
