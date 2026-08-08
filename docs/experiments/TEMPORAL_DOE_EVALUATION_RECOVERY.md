# Temporal DOE evaluation-only operational recovery

Date: 2026-08-08

Status: prospective recovery amendment; no candidate evaluation metrics existed
when this amendment was written

## Failure record

The frozen training study
`temporal-doe-seed1234-20260807-f638b49-v4` completed all five calibration
runs and all five 5,000-update primary runs.  Slurm job `504881` completed with
exit code `0:0`, and `training_receipts.json` was frozen before evaluation.

Evaluation job `504932` failed in the first arm before any sampler call or
metric row.  All ranks raised a CUDA out-of-memory error while
`torch.distributed.all_gather_object` exchanged the CPU action bank over the
default NCCL process group.  No arm summary, per-clip metric file, selection,
or protected-test access was produced.  The original study and its failure
state remain immutable.

After the CPU-collective repair passed that point, recovery job `504954`
exposed a second dormant evaluator-only typo while validating the first
checkpoint: two checks looked up `CHECKPOINT_SCHEMA` on the temporal trainer
module rather than on the shared semantic-screen/checkpoint module that owns
the constant.  This also occurred before a sampler call or metric row.  The
repair changes only those two validation references to the already frozen
shared schema constant.

Recovery job `504957` then reached the full checkpoint/config receipt audit
and failed because three historical file receipts were compared by absolute
checkout path, despite the evaluator's explicit clean-descendant compatibility
contract.  The bytes and Git objects were identical; only the clean checkout
root differed.  The amended audit now reopens both historical and current
regular files, requires identical SHA-256 and byte size, and permits only the
active dataset source path inside the producer attestation to change.  Every
other attestation field remains exact.  This failure also preceded every
sampler call and metric row.

## Minimal repair

Only the evaluator's metadata exchange changes.  Each rank still reads its
same manifest-strided action shard and performs the same shape, dtype,
finiteness, clip-ID, coverage, and order checks.  The serialized action shards
are exchanged through a temporary Gloo subgroup, keeping this CPU-only object
collective off CUDA.  The trained model, EMA checkpoint, sampler, corruption,
NFE grid, controls, metrics, analyzer, and all inference-critical files are
unchanged.

## Recovery policy

1. Commit the repair before any candidate evaluation metric exists.
2. Create a fresh clean detached checkout and a fresh implementation
   registration whose identity binds the repaired evaluator.
3. Write evaluation and analysis to a new recovery study ID.  Reference the
   original immutable calibration and primary checkpoints; do not retrain or
   replace them.
4. Preserve identical seed `20260801`, all 890 validation clips, NFE grid
   `1,2,4,8,12,20,25`, all nine controls, eight B200 ranks, and the frozen
   analysis gates.
5. Record the failed job, original training-receipt SHA-256, repair commit,
   recovery registration, and output receipts together.  Do not access the
   protected test split unless the unchanged development gate selects exactly
   one candidate.

This recovery can establish the originally preregistered scientific result;
it cannot be presented as an independently replicated seed.
