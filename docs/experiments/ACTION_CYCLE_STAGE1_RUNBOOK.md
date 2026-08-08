# Action-cycle Stage-1 runbook (disabled by Stage-0 STOP)

## Current decision

Do not run this study. The only eligible Stage-0 artifact returned
`stop_or_revise_action_cycle_path`; `tools/action_cycle_stage1.py register`
requires the opposite GO identity and will reject it. No manual override is
provided.

The draft resource estimate, had the prerequisite passed, was two parallel
single-node 8xB200 arms, approximately 35–50 minutes per arm including val64
aligned/shuffled/zero NFE-1 plus aligned NFE-4, followed by less than five
minutes of CPU analysis. A two-hour `short` QOS allocation per arm would have
been sufficient. This estimate is operational planning only, not measured
Stage-1 runtime.

## Conditions for reconsideration

Reconsideration requires a new, independently preregistered Stage-0 feature or
critic that passes all original recoverability controls, especially:

- better normalized MSE than the train mean;
- a material, simultaneous advantage over the same-clip nonoverlapping temporal
  negative;
- positive retrieval lower bound on future-relevant transitions; and
- no use of protected test or post-hoc validation subset selection.

If such a new Stage 0 passes, create a new Stage-1 branch and registration. Do
not reuse this validation gate as though it were confirmatory: its design is
now public and any future val64 use is adaptive exploratory evidence. Re-audit
the code, run the deployment canary, and scheduler-dry-run the Slurm scripts
before submission. The present prototype must never be launched against the
failed Stage-0 ridge.
