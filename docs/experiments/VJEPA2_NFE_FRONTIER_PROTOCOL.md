# V-JEPA 2.1 NFE-frontier repair

The original controlled-study speed gate compared J1@4 with VPM@8. That
immutable v3 result does not establish acceleration because the inspected test
grid showed that VPM@8 was not necessarily the best quality/compute baseline.
This opt-in protocol selects a conservative baseline on validation, freezes one
candidate, evaluates it once on a newly constructed lockbox, and benchmarks the
three required endpoints on one B200.

This protocol does not alter the active v3 contract or reinterpret its existing
test evidence as held out.

## Reproducible Slurm orchestration

The complete cluster workflow is encoded in
`tools/slurm/submit_vjepa2_frontier_workflow.sh`. Its checked-in defaults bind
the v3 study, update-1000 array job `481132`, immutable cache build, LACWM
Python, V-JEPA extractor Python, official source/checkpoint, and train-only
PCA. Its scheduler defaults match the active study: partition `batch`, account
`coreai_chef_posttrain`, and QOS `normal`; each remains CLI-overridable. Keep
the active training checkout at `9cf8e69` while the original jobs run.
Fetching the evaluator commit and creating a detached evaluator worktree does
not change that checkout:

```bash
BASE=/lustre/fsw/portfolios/coreai/projects/coreai_chef_pretrain/users/ldu/lacwm_train
TRAIN_REPO=$BASE/src/vjepa2-latent-forcing/dual-video-diffused-lacwm
git -C "$TRAIN_REPO" fetch github research/vjepa2-latent-forcing-study
EVALUATOR_COMMIT="$(git -C "$TRAIN_REPO" rev-parse github/research/vjepa2-latent-forcing-study)"
EVALUATOR_REPO=$BASE/src/vjepa2-frontier-evaluator-${EVALUATOR_COMMIT:0:12}
git -C "$TRAIN_REPO" worktree add --detach "$EVALUATOR_REPO" "$EVALUATOR_COMMIT"

"$EVALUATOR_REPO/tools/slurm/submit_vjepa2_frontier_workflow.sh"

# After reviewing the read-only preflight:
"$EVALUATOR_REPO/tools/slurm/submit_vjepa2_frontier_workflow.sh" \
  --evaluator-commit "$EVALUATOR_COMMIT" \
  --execute
```

The initial submission queues one-B200 lockbox extraction/registration and an
update-1000 artifact gate behind the final training array, followed by separate
eight-B200 VPM/J1 validation jobs. The selection job independently reproduces
the raw-row selection. It does not submit or even create pending lockbox-scoring
jobs when
`confirmatory_eligible=false`. An eligible selection dynamically creates the
remaining `afterok` chain: a two-task VPM/J1 lockbox array, confirmation,
one-B200 timing, and finalization.

At execution, the launcher queries Slurm accounting for the recorded u1000
array using formatted array-task IDs. It expands Slurm's compressed active
rows and requires the exact task set `481132_0` through `481132_4`, once each.
While any task is pending/running, cache and artifact-gate jobs use
`afterok:481132`. Only when every one of those five allocations reports
`COMPLETED/0:0` does the launcher omit that controller dependency—which may be
stale after `MinJobAge`—and immediately run the same full final-artifact gate.
Failed, missing, duplicate, extra, or ambiguous accounting aborts submission.
The unrelated original paired job `481133` may finish independently against
the untouched training checkout.

Every scientific output is fresh-only. The launcher refuses any pre-existing
frontier output tree, freezes one clean evaluator commit that is an
inference-tree-identical descendant of the training commit, and verifies that
the official V-JEPA source checkout has not changed. It also requires that the
evaluator worktree be distinct from the untouched, clean training checkout
recorded in the study. The timing entrypoint creates the fresh
`frontier_latency` parent before invoking its exclusive writer.

## Evidence sequence

1. Construct a fresh 128-clip lockbox from the pinned source episode
   population. The construction excludes every episode in the original
   train/validation/inspected-test manifests and deterministically takes the
   next eligible episodes under seed `20260730`, one clip per episode.
2. Extract its V-JEPA cache with the original train-only PCA. Register the
   lockbox after full manifest/cache/target/RGB/action hashing and explicitly
   attest that it has never been scored, previewed, or used for model/NFE
   selection.
3. Evaluate autonomous VPM and J1 at NFE `1,2,4,6,8,12,20` on all 64 pinned
   validation clips.
4. Bind the already registered lockbox into the validation selection. Freeze
   one pair `J1@k` versus `VPM@m`, with `2 <= k < m`.
5. Evaluate only J1@k, VPM@k, and VPM@m on all 128 lockbox clips. Every row
   binds both the selection and lockbox identities. No lockbox metric may alter
   the selected pair.
6. Confirm paired reconstruction gates, then benchmark J1@k, VPM@k, and VPM@m
   in one process with both models resident on one B200.
7. Finalize only after recomputing endpoint, quality, timing, lockbox, stage,
   and Git-provenance bindings. Selection and confirmation JSONL inputs are
   reloaded and fully rehashed; the validation winner and both lockbox
   comparisons are reproduced from those raw rows. A self-rehashed posthoc or
   aggregate-only artifact cannot be promoted to a confirmatory result.

NFE=1 remains useful when constructing the VPM compute/quality curve, but it is
never a J1 causal candidate: its sole Wan call sees the initial auxiliary noise,
not a denoised V-JEPA state.

## Selection and gates

The burden of proof is on extra baseline compute. A higher-NFE VPM point enters
the conservative frontier only if it passes all three paired validation checks
against every lower-NFE VPM point:

- temporal-difference MSE relative-improvement 95% CI-low is strictly positive;
- video-latent NMSE CI-low is greater than -1%; and
- decoded RGB MSE CI-low is greater than -1%.

Statistically unresolved VPM comparisons therefore fall back to lower compute;
they cannot inflate the apparent J1 call reduction. In addition, a validation
candidate must pass the same-NFE J1@k versus VPM@k attribution gate: temporal
CI-low at least 3% (the original T1 threshold), with both guardrail CI-lows
above -1%.

Candidate ordering is fixed: maximum NFE reduction, frontier temporal CI-low,
same-NFE attribution temporal CI-low, minimum guardrail CI-low, lower J1 NFE,
then lower VPM NFE. This optimization uses validation only.

The lockbox repeats both comparisons without reselection:

- J1@k versus VPM@k establishes V-JEPA attribution; and
- J1@k versus VPM@m establishes quality-preserving call reduction.

Timing requires a paired speedup CI-low above zero, lower J1 p95, and favorable
mean latency in both pairwise execution-order strata. The 120 timed rounds form
20 complete six-permutation counterbalance blocks; bootstrap resampling uses
those complete blocks to preserve order balance and thermal autocorrelation.
The artifact separately reports same-NFE overhead, p50/p95/mean, calls, decoded
frames/s, endpoint/audit/resident peak memory, order strata, and the B200
identity.

## 1. Construct and register the lockbox

All outputs and caches belong on durable `/mnt/data1`, `/mnt/data2`, or the
study's Lustre allocation. Start from the clean repair commit:

```bash
python tools/vjepa2_frontier_lockbox.py build-manifest \
  --repo-root REPO_ROOT \
  --registration-commit EVALUATOR_COMMIT \
  --study-manifest STUDY_ROOT/study_manifest.json \
  --output-dir STUDY_ROOT/frontier_lockbox
```

The builder is fail-closed and refuses an existing output directory. It records
the pinned source episode-manifest hash, ex-ante ranking rule/seed, all original
split episode-set digests, and exact zero-overlap counts.

Use the existing extractor and the study's original training manifest/PCA:

```bash
EXTRACTOR_PYTHON tools/extract_vjepa2_targets.py extract \
  --split test \
  --clip-manifest STUDY_ROOT/frontier_lockbox/lockbox.jsonl \
  --train-manifest ORIGINAL_TRAIN_MANIFEST \
  --pca ORIGINAL_TRAIN_ONLY_PCA \
  --cache-dir STUDY_ROOT/frontier_lockbox/cache \
  --source-path PINNED_VJEPA_SOURCE \
  --checkpoint PINNED_VJEPA_CHECKPOINT \
  --checkpoint-sha256 PINNED_VJEPA_CHECKPOINT_SHA256 \
  --device cuda \
  --batch-size 1
```

`split=test` above is only the physical label accepted by the existing cache
extractor. All downstream evidence uses semantic split `lockbox`.

After extraction, register it before any lockbox evaluation:

```bash
python tools/vjepa2_frontier_lockbox.py register \
  --repo-root REPO_ROOT \
  --registration-commit EVALUATOR_COMMIT \
  --study-manifest STUDY_ROOT/study_manifest.json \
  --construction STUDY_ROOT/frontier_lockbox/lockbox_construction.json \
  --cache-metadata STUDY_ROOT/frontier_lockbox/cache/metadata.json \
  --attest-never-scored \
  --output STUDY_ROOT/frontier_lockbox/registration.json
```

Registration re-derives the deterministic next-unused sample, rechecks physical
episode identities, hashes every cache array, verifies exact shapes/dtypes, and
requires the original train-manifest, PCA, V-JEPA checkpoint, and source
commit.

## 2. Validation selection

Run the eight-rank evaluator once for each final arm:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  tools/evaluate_vjepa2_quality.py \
  ...existing final-checkpoint arguments... \
  --expected-commit TRAINING_COMMIT \
  --evaluator-commit EVALUATOR_COMMIT \
  --frontier-split validation \
  --output-dir RUN_DIR/frontier_quality/validation/update_1000
```

Freeze the candidate while binding the unscored lockbox:

```bash
python tools/vjepa2_nfe_frontier.py select \
  --j1-rows J1_RUN/frontier_quality/validation/update_1000/rank_*.jsonl \
  --vpm-rows VPM_RUN/frontier_quality/validation/update_1000/rank_*.jsonl \
  --lockbox-registration STUDY_ROOT/frontier_lockbox/registration.json \
  --output STUDY_ROOT/frontier_selection.json
```

## 3. One-time lockbox quality

Run each arm once, using the frozen selection:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  tools/evaluate_vjepa2_quality.py \
  ...existing final-checkpoint arguments... \
  --expected-commit TRAINING_COMMIT \
  --evaluator-commit EVALUATOR_COMMIT \
  --frontier-split lockbox \
  --frontier-selection STUDY_ROOT/frontier_selection.json \
  --output-dir \
    RUN_DIR/frontier_quality/lockbox/LOCKBOX_SHA/SELECTION_SHA/update_1000
```

The evaluator clones the pinned `viz_dataset` config and overrides only
`datasets.ABC.clip_manifest` and `datasets.ABC.cache_metadata`. Rank 0
re-derives the construction and fully rehashes all lockbox arrays. All ranks
remain augmentation-free and teacher-free, with batch two on eight B200s.

```bash
python tools/vjepa2_nfe_frontier.py confirm \
  --selection STUDY_ROOT/frontier_selection.json \
  --j1-rows J1_LOCKBOX_DIR/rank_*.jsonl \
  --vpm-rows VPM_LOCKBOX_DIR/rank_*.jsonl \
  --output STUDY_ROOT/frontier_lockbox_confirmation.json
```

## 4. Same-B200 timing and finalization

Inside a one-GPU B200 Slurm allocation:

```bash
python tools/benchmark_vjepa2_frontier_latency.py \
  --repo-root REPO_ROOT \
  --expected-commit TRAINING_COMMIT \
  --benchmark-commit BENCHMARK_COMMIT \
  --study-root STUDY_ROOT \
  --selection STUDY_ROOT/frontier_selection.json \
  --output STUDY_ROOT/frontier_latency/paired.json

python tools/vjepa2_nfe_frontier.py finalize \
  --selection STUDY_ROOT/frontier_selection.json \
  --confirmation STUDY_ROOT/frontier_lockbox_confirmation.json \
  --latency STUDY_ROOT/frontier_latency/paired.json \
  --output STUDY_ROOT/frontier_final_report.json
```

`TRAINING_COMMIT` remains the commit bound to checkpoints.
`EVALUATOR_COMMIT`/`BENCHMARK_COMMIT` must be clean descendants whose complete
LACWM/modeling/dataset inference trees are Git-object-identical to the training
commit. Only tools, tests, or documentation may differ.

## Claim limits

A PASS means faster inference with better held-out paired reconstruction for
this one seed-1234 checkpoint pair: temporal-difference MSE improves with
video-latent NMSE and decoded RGB MSE within their margins, and the paired B200
timing gate passes. It is not an FVD, perceptual-quality, diversity, or general
video-generation-quality result.

The lockbox is unseen by controlled fine-tuning and NFE selection. The shared
LACWM warm-start may have been pretrained on ABC, potentially including these
episodes, unless separate pretraining provenance proves otherwise. Clip
bootstrap quantifies sample uncertainty, not training-run variance; multi-seed
retraining is required before a paper-level method-generalization claim.
Timing p95 is the distribution of 120 counterbalanced repeats of one fixed
lockbox clip, not a latency distribution across diverse clips.

The already inspected v3 test grid is posthoc only. It may be analyzed with
`select --split test --expected-clips 128 --allow-posthoc`, but it cannot be
registered as the lockbox, cannot produce `confirmatory_eligible=true`, and
cannot be finalized into a PASS.
