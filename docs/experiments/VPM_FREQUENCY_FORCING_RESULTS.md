# VPM generated frequency-forcing screen: results

Date completed: 2026-08-08

This prospective four-arm screen tested whether a compact, per-view Haar
DC-plus-motion stream can be learned and generated from noise early enough to
improve the frozen ABC video-latent world model.  It used one seed, 200 matched
continuation updates, all 64 validation clips, NFE `{1,2,4,8}`, and no
protected-test access.  Every auxiliary bin started from noise at deployable
inference.

The complete artifact is:

`frequency_forcing/frequency-forcing-seed1234-20260808-8510b95-v3`

- registration identity:
  `f9df7f1496f18394d411b39183f067a37c547ae74705db369d8af98b6079dc90`;
- analysis identity:
  `32a9af9def2e0e2a86c8bd7a216b4c7b79603d073c8c03e7f639f7ca86e1088f`;
- analysis SHA-256:
  `69cad0af40199ef6cb5102252db959b763a7a141d993a9ae5f4e14560c8a990c`;
- audited validation rows: `5,120`;
- passing candidates: none;
- protected test accessed: false.

## Arms

| Arm | Auxiliary objective | Video sees generated TF state | TF clock |
|---|---|---|---|
| `FPM` | no | no | none |
| `FAUX` | yes | no | synchronous |
| `FSYNC` | yes | yes | synchronous |
| `FLEAD` | yes | yes | earlier-maturing |

The auxiliary state was `[B,6,4,24,120]`: three per-view Haar DC channels and
three temporal-motion channels on the Wan latent grid.  The design did not
pass RGB future frames, cached features, or a clean auxiliary target to the
deployable sampler.

## Primary NFE-1 result

Lower is better.

| Arm | video latent NMSE | decoded MSE | temporal MSE | relative to `FPM` |
|---|---:|---:|---:|---|
| `FPM` | 0.239028 | 0.0189623 | 0.0132683 | reference |
| `FAUX` | 0.254608 | 0.0198024 | 0.0134659 | -6.52% / -4.43% / -1.49% |
| `FSYNC` | 0.254845 | 0.0199012 | 0.0134657 | -6.62% / -4.95% / -1.49% |
| `FLEAD` | 0.254304 | 0.0196939 | 0.0133660 | -6.39% / -3.86% / -0.74% |

All three auxiliary-training arms regressed the video-only control.  Their
95% paired intervals exclude zero for the NFE-1 latent and decoded
regressions.  `FLEAD` was modestly better than `FSYNC`, but this only recovered
a small fraction of the multitask penalty and did not beat `FPM`.

Within the trained joint checkpoints, generated-state fusion did not establish
sample-specific guidance:

- `FSYNC` autonomous versus its own fusion-off endpoint changed temporal MSE
  by +0.244%, while latent NMSE and decoded MSE worsened by 0.625% and 0.589%;
- `FLEAD` autonomous versus fusion-off changed temporal MSE by +0.264% and
  decoded MSE by +0.349%, while latent NMSE worsened by 0.190%;
- aligned and globally shuffled generated states were bit-identical at NFE 1
  and differed by at most about 0.13% at later endpoints.

The NFE-4 `FLEAD` temporal point effect versus `FPM` was only +0.409%, paired
with 3.17% worse latent NMSE and 4.51% worse decoded MSE.  NFE 8 also failed.
No candidate satisfied the preregistered quality, guardrail, and alignment
gates.

## Interpretation

The auxiliary prediction loss itself caused most of the damage: `FAUX`, which
never fused the TF state into video, was already substantially worse than
`FPM`.  Adding synchronous or leading fusion changed video quality only
slightly relative to `FAUX`.  The learned state gate also shrank during
training, consistent with the video branch suppressing an unhelpful generated
signal.

At NFE 1, a synchronous joint forward cannot use the auxiliary clean estimate
produced by that forward: Wan observes auxiliary noise, and both state updates
happen only after the call.  An earlier auxiliary clock cannot repair that
within-call causal ordering.  At later NFE, the nearly zero aligned-versus-
shuffled effect shows that the generated TF state still did not become a
sample-specific motion guide.

This rejects this particular fixed Haar representation, training budget, and
synchronous shared-Wan mechanism.  It does not prove that every frequency
representation or every dual-video architecture is impossible.  It instead
motivates the next two controlled mechanisms: generate a small causal plan
*before* Wan, or predict and inject an auxiliary state between early and late
Wan blocks in the same call.
