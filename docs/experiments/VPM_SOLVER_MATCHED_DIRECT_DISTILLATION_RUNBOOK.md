# SMTD-4 P0 runbook

Status: source preparation only. There is no registration or submission path.
Readiness reports `READY_SOURCE_BLOCKED_EXECUTION_REVIEW` because the current
parent is not a quality-dominant multi-step teacher.

## Current safe operation

From a clean exact implementation commit, create a read-only readiness seal:

```bash
python tools/low_nfe_direct_baseline.py readiness \
  --source-repo "$PWD" \
  --expected-commit "$(git rev-parse HEAD)" \
  --output /approved/analysis/root/readiness_seal.json
```

This command succeeds only as an audit operation and writes a blocked execution
decision. It never creates a registration, launches Slurm, opens data, or
contacts W&B.

Inspect the immutable resource proposal without writing anything:

```bash
python tools/low_nfe_direct_baseline.py plan
```

## Preconditions for any future execution

1. The exact source commit is clean, pushed, and reachable from the audited
   GitHub remote.
2. The exact test report and readiness seal bind that same 40-character commit
   and tree.
3. A human explicitly accepts that the current parent is not a dominant
   multi-step teacher and authorizes a bounded local-consistency test after
   reviewing the 112 B200-hour and 100 GiB ceilings.
4. A three-copy, single-rank real-model preflight passes the memory ceiling.
5. A separate launcher/registration change is reviewed. This source contains
   no Slurm submission path by design.

## Forbidden shortcuts

- Do not use a privileged TF/V-JEPA/clean-video teacher.
- Do not change NFE, stride, boundary scaling, EMA, target mixing, updates, or gates after
  seeing validation outcomes.
- Do not serialize the frozen teacher into a student checkpoint or instantiate
  it in the deployment evaluator.
- Do not enable W&B in the prepared Hydra configs.
- Do not open the protected test split for development selection.
