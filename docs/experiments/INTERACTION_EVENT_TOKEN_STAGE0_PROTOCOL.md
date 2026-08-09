# Explicit interaction-event token fallback

Date frozen: 2026-08-08

Status: **prospective sequential fallback; no token outcome has been scored**

## Why this fallback exists

The preregistered masked-event PCA32 screen stopped for two separable reasons:
aligned actions did not improve decoded event-field prediction beyond history,
and the learned dense-map PCA transferred only 4.52% better than a fit-mean
field on held-out episodes. This final bounded fallback asks whether learned
spatial-subspace instability hid a simpler action-predictable interaction-rate
signal.

The earlier decision remains immutable. This is a new study designed after
that result, so it cannot be presented as confirmatory evidence for the PCA32
hypothesis.

## Fixed population and leakage boundary

Reuse exactly the registered fit256/score64 D405 train episodes, motion strata,
and episode-shuffled donor map from registration
`f985c94c708950a6e90530044a9b91f58e539a50a3f7ac2f91b89b1dc945f393`.
No episode, donor, action window, or mask parameter is reselected. Validation
and protected test remain unsupported.

Inputs and target-only boundaries are unchanged:

- causal input: top RGB frames 0--4, measured history state for observed robot
  masks, and planned action chunks 4--11;
- target-only: future RGB/state boundaries 5--12;
- same official YAM/D405 render, 12-pixel union-mask dilation, 45x80 work grid,
  thresholds, proximity weighting, and four raw event-field channels.

## Explicit 32-scalar state

For each of eight future transitions, spatially average the four masked event
channels:

1. positive luminance-change mass;
2. negative luminance-change mass;
3. signed horizontal event-weighted transport;
4. signed vertical event-weighted transport.

The result is `[8,4]`, flattened to 32 scalars. There is no learned target PCA
or pretrained encoder. Positive plus negative mass exactly recovers each
horizon's weighted photometric-event integral; the two signed tokens exactly
recover the corresponding transport integrals. This eliminates target-encoder
generalization error, but deliberately discards spatial localization and object
identity.

Observed history uses the analogous `[4,4]` 16-scalar token. Planned actions
remain fit-standardized PCA32. Two ridge models have equal 48-wide designs:

- history-only: `[history16, zeros32]`;
- history+action: `[history16, action_PCA32]`.

Both use five-fold fit-only alpha selection over
`{0.1,1,10,100,1000,10000}` and predict train-standardized future tokens.

## Controls and fixed gate

Evaluate aligned, same-stratum episode-shuffled, train-mean, raw-zero, and
nonwrapping -1/+1 action windows. Raw zero and shifts are diagnostic.

The three mandatory all-token comparisons—aligned versus history-only,
episode-shuffled, and train-mean—share one common 10,000-draw score-episode
bootstrap with seed 20260814. One-sided Bonferroni lower bounds use confidence
`1 - 0.05/3 = 98.3333%`. Every comparison must show:

- at least 5% lower standardized token MSE;
- simultaneous lower bound strictly above zero;
- at least 60% favorable score episodes.

Change-only and signed-transport-only effects are mandatory reported
diagnostics but do not replace the all-token gate. The target salience check
requires at least 24/32 fit token dimensions to have raw standard deviation
above `1e-6`, and at least 90% of score clips to contain total positive plus
negative event mass above `1e-5`.

A pass yields only `GO_FOR_SPATIAL_LOCALIZATION_SCREEN`: global integrals are
not spatially sufficient for Wan conditioning, so this study can never directly
authorize generator integration. Any later spatial screen must predict a
location-bearing state and establish its own relevance gate. A failure yields
`STOP_EXPLICIT_EVENT_TOKENS` and ends this interaction-event line before video
training.

## Claim boundary

This is a train-only, one-seed linear predictability screen. It does not measure
video quality, semantic objects, contact, forces, multimodal futures, control
rollouts, or generator acceleration. Reusing the same score episodes makes it a
sequential diagnostic, not independent confirmation.
