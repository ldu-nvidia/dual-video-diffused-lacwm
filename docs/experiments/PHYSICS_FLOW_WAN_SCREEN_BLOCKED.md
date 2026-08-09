# Fixed causal robot-flow Wan scaffold: blocked handoff

Date: 2026-08-08

Status: **`BLOCKED_INCOMPLETE`; no training or video evaluation was launched**

## Decision

The fixed-field Wan scaffold must not be treated as an executed experiment.
Its frozen prerequisite, the predicted tracking-corrected renderer attribution
gate, returned `STOP_RENDERER_ATTRIBUTION`: corrected robot-flow EPE improved
only 3.16% over raw commands, its paired interval crossed zero, and only 13/24
clips favored correction. The protocol requires a positive interval, at least
5%, and at least 60% favorable clips. Creating caches or launching Wan from
that trajectory would violate the prospective branch rule.

The code is retained because it fixes the intended model seam and tensor
contract, and can be completed without redesign if a fresh recurrent, hybrid,
or raw causal renderer passes an independently preregistered gate. It is not
an evidence-producing workflow yet.

## Implemented and tested

- Fixed field shape `[B,16,4,24,120]`, packing eight future transitions as
  four `(dx/W,dy/H,visibility,log-depth-ratio)` components into each of two
  future Wan temporal tokens.
- Exact-zero history and wrist-view support, exact pack/unpack round trip,
  nonwrapping time shift, tensor hashing, displacement bounds, and rejection
  of nonzero hidden components.
- A clean fixed-condition seam through the existing Wan adapter with no second
  clock, no field velocity loss, and no field-model call.
- An optional support mask that keeps truly unsupported patches exactly zero
  even after biased Conv3d/LayerNorm optimization.
- Runtime FLOW-OFF masking with a nonzero trainable gate; output, video loss,
  and shared gradients are invariant to the supplied field.
- Parent VPM history corruption is preserved: history follows its ordinary
  forward-noise trajectory during training and sampling instead of being
  prematurely clean-clamped.
- The ABC wrapper retains the existing ABC morphology identity.
- Training does not instantiate development RGB loaders; val64 is reserved for
  the later causal endpoint evaluator.
- Parent/output path separation and a checkpoint/trace resume design that does
  not overwrite restored learned weights with the warm-start parent.

Focused compatibility validation after these fixes passed 94 tests with two
optional-runtime skips. Python compilation and `git diff --check` passed.

## Missing before any launch

An independent read-only audit correctly found that the repository still lacks
the evidence workflow needed to support the protocol:

1. a causal cache builder and content-bound registration tying every field row
   to the ordered clip ID, RGB/action hashes, action window, camera order,
   D405 eligibility, predictor, calibration, geometry, renderer source, and
   passing analysis identity;
2. materialized and audited aligned, episode-shuffled, action-shuffled,
   nonwrapping time-shifted, and wrong-calibration controls;
3. an endpoint evaluator that materializes all causal samples before opening
   clean future RGB and separately times predictor, renderer, VAE, adapter,
   Wan, decoder, and complete serving path;
4. the preregistered LPIPS/reconstruction/action-sensitivity analyzer,
   clustered bootstrap, D405>=32 enforcement, decision receipt, and
   independent artifact auditor;
5. a guarded launcher that hashes the nonpersistent null prompt, scheduler
   configuration, resolved arm configs, initial auxiliary state, parent state,
   rank topology, optimizer, and paired data/noise traces; and
6. an explicit trace-comparison audit proving 200 matched OFF/ON training
   updates and safe preemption recovery.

The present dataset joins RGB/action and field by row index and checks the
global field hash, but it does not yet bind enough row-level lineage to reject
a permuted, differently packed, or self-asserted cache. The sampler also
accepts an external condition tensor plus a label; without the missing
evaluator, that label does not prove donor or calibration identity.

## Re-entry condition

Complete the missing workflow only after one fresh causal renderer family
passes its own gate. The recurrent increment predictor, absolute-anchor plus
raw-delta hybrid, and raw planned-geometry fallback are being evaluated on 24
previously untouched D405 episodes with global multiplicity correction. A
passing arm must then be frozen into a new cache registration; this stopped
predicted-ridge analysis may not be reinterpreted as permission to launch.
