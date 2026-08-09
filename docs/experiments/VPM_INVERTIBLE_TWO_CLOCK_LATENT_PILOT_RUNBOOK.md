# IPQ-TC1 sealed contingency runbook

## Terminal status

IPQ-TC1 is an implemented but **unlaunched contingency**. The upstream ILSF-2
handoff ended with `NO_GO_GENERIC_EARLY_SUBSPACE` and audit status `PASS`.
That decision does not satisfy the registration gate, which accepts only an
independently audited `ADVANCE_IPQ_TC1` receipt. Therefore:

- no IPQ registration exists;
- no training or endpoint job was submitted;
- no W&B run or write occurred;
- no validation endpoint row or outcome was opened; and
- no source was staged to the cluster for execution.

The additive package is retained so the scientific design and causality fixes
are reproducible if a future, separately frozen study revisits this direction.
It must not be launched under the present ILSF-2 decision.

## Package boundaries

The package adds a view-isolated spatial Haar-LL projector `P`, exact
complement `Q=I-P`, independent/tied clock construction, hard projected
velocity loss, a dedicated `InvertibleTwoClockVPM`, a strict matched trainer,
target-blind endpoint evaluator, paired analyzer, parent-parity comparator,
registration/readiness tool, two non-requeueable Slurm entrypoints, and focused
tests. It does not edit the production LACWM model, Wan forward model, adapters,
or production sampler.

The endpoint evaluator never constructs the production ABC dataset. Before a
global eight-rank barrier, its typed reader can serve only `RGB[row,0:5]` and
planned `actions[row,0:13]`. Every endpoint is generated, decoded, call-counted,
copied, and hashed first. Full RGB access and whole-array integrity rehashing
are available only to the distinct post-barrier scoring reader. Per-rank access
ledgers preserve the exact file, row, slice, tensor hash, and event order.

## Read-only seal

After the exact implementation commit is pushed, readiness is the only
applicable command under the terminal handoff:

```bash
python tools/invertible_two_clock_pilot.py readiness \
  --source-repo /absolute/path/to/clean/worktree \
  --expected-commit <exact-40-character-commit> \
  --remote origin \
  --test-report /absolute/path/to/exact-source-test-report.json \
  --ilsf-handoff-decision NO_GO_GENERIC_EARLY_SUBSPACE \
  --ilsf-audit-status PASS
```

The seal must report remote reachability, parent/source/config hashes, required
file inventory, forbidden production-file diff intersection `[]`, test receipt,
resource envelope, `registration_created=false`, `jobs_submitted=0`,
`wandb_writes=0`, `outcomes_opened=0`, and `registration_gate_open=false`.

## Historical/current parent gate

No future registration may attribute a source regression to IPQ. Independent
auditors must run the untouched `de65…` parent in clean historical
`6560866…` and current worktrees on the same target-blind training-history
replay. Their NFE 1/2/4 receipts are compared by
`tools/invertible_two_clock_parent_parity.py`; initial noise, final latent, and
decoded uint8 hashes must all match bit-for-bit. The comparison cannot open
validation or protected-test arrays.

## Hypothetical resource ceiling

The dormant resource plan is two one-node/eight-B200 training allocations of
at most four hours, two one-node/eight-B200 endpoint allocations of at most two
hours, and CPU-only analysis: at most 96 reserved B200-hours. Both Slurm files
require a clean exact source SHA, fresh registered Lustre paths, at least 1 TiB
free, one node/eight B200s, and no requeue. They do not submit other jobs. These
guards describe a possible future protocol; they do not override the present
terminal stop.
