# Prospective object/contact-slot Stage-0 protocol

Date frozen: 2026-08-08

Status: **prospective; no fresh score RGB/state or outcome has been opened**

## Question and sequential-study boundary

The dense masked-event PCA32 and explicit global-integral token studies both
stopped. This new study tests a different hypothesis: planned robot action may
predict a sparse, location-bearing interaction state even when it cannot
predict a dense change field or global transport integral.

The representation was designed after viewing the earlier score outcomes, so
those 64 episodes are permanently excluded. Reusing their outcome would be
post-selection. The earlier 256 predictor-fit episodes may be reused only for
model fitting; a deterministic fresh score set is selected from D405 episodes
that belonged to neither parent fit nor parent score. The split is sealed
before any selected RGB pixel or measured state array is indexed. Validation
and protected test are unsupported.

## Frozen split construction

Recompute the original 415-episode D405 eligibility scan and four planned-joint
motion strata from immutable ABC train512 metadata and cached actions. Require
the recomputed parent fit256/score64 indexes and strata to match registration
`f985c94c708950a6e90530044a9b91f58e539a50a3f7ac2f91b89b1dc945f393`.

Remove all 320 parent-selected episodes. Within each remaining original motion
stratum, rank untouched clips by SHA-256 of
`clip_id|object-contact-slot-fresh-score-v1`; select the first 16, for 64 fresh
score episodes total. Rotate those 16 within stratum to assign the shuffled
action donor. Store the full untouched candidate pool, ranks, exclusions, and
selection in registration. No result may be computed until registration and
the exact source commit are reported.

## Severe causal slot representation

Inputs are top RGB frames 0--4 and planned action chunks 4--11. Future RGB and
measured state at frames 5--12 are target-only. Measured state renders the
official YAM silhouette but is never a predictor feature.

For each adjacent-frame transition:

1. Render the source/target articulated robot, union the masks, and dilate by
   16 pixels at 180x320 before downsampling to 45x80.
2. Compute thresholded positive/negative luminance change and Farneback flow.
3. Retain connected support only where change is at least 0.20, flow magnitude
   is at least 0.20 work-grid pixels, and the pixel is outside the robot mask.
   Close with a deterministic 3x3 kernel; reject components below four pixels,
   above 20% of the image, or farther than 12 work pixels from rendered robot
   support.
4. Rank at most 12 candidates by robot-proximity-weighted event mass. Causally
   match them into four persistent slots using predicted-centroid distance and
   a mass penalty. Matching can inspect only the current and earlier
   transitions, tolerates one missed transition, and uses deterministic
   slot/candidate tie breaks. Unobserved slots are exactly zero.

Each slot has 14 explicit coordinates: presence, normalized centroid x/y,
area, positive/negative mass, 2D covariance xx/xy/yy, signed mean flow u/v,
robot-proximity contact score, positive contact-score onset, and track age.
The history tensor is `[4,4,14]`; the future target is `[8,4,14]`. Persistent
slot index is the identity carrier. There is no learned target encoder and no
future-informed backward matching.

## Predictor and controls

Flatten history to 224 scalars and concatenate fit-standardized action PCA32.
The equal-width history-only model receives the same history plus zeros32.
Separate ridge models select alpha from `{0.1,1,10,100,1000,10000}` by
five-fold fit256-only target-standardized MSE. Only target coordinates with fit
standard deviation above `1e-6` are modeled.

Evaluate aligned action against all six frozen references:

- history-only model;
- same-stratum fresh-episode-shuffled action;
- parent-fit mean raw action;
- all-zero raw action;
- nonwrapping action window shift -1;
- nonwrapping action window shift +1.

Two causal metrics are mandatory: active-coordinate standardized slot MSE and
decoded field MSE. Slot decoding uses covariance-clipped Gaussian ellipses that
preserve each slot's positive/negative mass and signed mean flow. All 12 causal
contrasts share a 10,000-draw fresh-episode bootstrap, seed 20260815, and
one-sided Bonferroni confidence `1 - 0.05/12 = 99.5833%`.

Every contrast must show at least 5% aligned improvement, a strictly positive
simultaneous lower bound, and at least 60% favorable episodes. Thus a raw-zero
or shuffled win cannot compensate for failure against history, mean, or either
timing shift.

## Relevance and salience gates

Decode oracle target slots and compare their cleaned component-field MSE with
the fit256 mean field and all-zero field. Both comparisons use a common
episode bootstrap with one-sided Bonferroni 97.5% lower bounds and must achieve
at least 20% improvement, a positive lower bound, and 60% favorable episodes.

Additionally require all of:

- at least 50% of the 448 future target coordinates active on fit256;
- at least 90% of fresh score clips with a nonnull future slot;
- at least 60% with some slot present in two adjacent future transitions.

Only if every causal, timing, reconstruction, and salience gate passes may the
decision be `GO_FOR_OBJECT_SLOT_GENERATOR_SCREEN`. Otherwise it is
`STOP_OBJECT_CONTACT_SLOT`. No Wan run, validation/test access, video-quality
claim, or control claim is permitted in this Stage-0 study.

## Claim boundary

Connected motion components and proximity are observational object/contact
proxies, not semantic instance masks or force labels. The fresh score supports
one-seed train-population predictability and relevance only. Bootstrap
uncertainty covers score-episode sampling, not retraining variance.
