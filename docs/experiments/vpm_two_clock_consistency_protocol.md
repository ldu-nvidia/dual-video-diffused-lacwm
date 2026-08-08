# VPM two-clock self-consistency screen (prospective)

## Question and claim boundary

This protocol is fixed before either arm is trained or scored. It asks whether
a deployable video generator can learn a better high-noise velocity field by
matching its high-noise clean prediction to its own lower-noise clean
prediction during training. The auxiliary signal is another prediction of the
same video model, not a clean future feature, pretrained encoder, or inference
condition.

The experiment is a 512-train/64-validation ABC continuation screen. The
protected 128-clip test split is not configured or opened. A pass is evidence
for this one seed and short continuation only; it is not a test-set,
real-time-latency, or physical-realism claim.

## Rectified-flow construction

LACWM uses `sigma=1` for noise and `sigma=0` for clean data:

```text
x_sigma = (1-sigma) x0 + sigma epsilon
v*      = epsilon - x0
x0_hat  = x_sigma - sigma v_theta(x_sigma, sigma, conditioning).
```

Each training sample draws one Gaussian `epsilon`. Two native scheduler points
are then sampled uniformly from the discrete points in the fixed bands:

```text
sigma_hi in [0.8, 1.0]
sigma_lo in [0.0, 0.4].
```

Both noisy states lie on the same straight RF trajectory:

```text
x_hi = (1-sigma_hi) x0 + sigma_hi epsilon
x_lo = (1-sigma_lo) x0 + sigma_lo epsilon.
```

The Wan video model is called twice, low clock first and high clock second. Both
predictions receive standard future-only RF supervision:

```text
L_flow = 0.5 * [MSE(v_hi, epsilon-x0) + MSE(v_lo, epsilon-x0)].
```

Let `M` be the existing future-video validity mask. The candidate also uses
the stopped low-noise clean prediction as a training target for the high-noise
clean prediction:

```text
E_b = mean_M(x0_b^2) + 1e-6

L_cons = mean_b mean_M(
           [x0_hat_hi - stopgrad(x0_hat_lo)]^2
         ) / stopgrad(E_b)

L_total = L_flow + lambda_cons L_cons.
```

The clean latent appears only in the scale denominator and in the ordinary RF
target already required for diffusion training. It is never a condition to
the transformer. Gradient from `L_cons` reaches the high-noise prediction only;
the low-noise prediction is a stopped self-teacher.

## Frozen parent and matched arms

Both arms strictly load every key from the update-1,000 parameter-matched
video-only VPM snapshot in
`vjepa2-controlled-20260730-seed1234-9cf8e69-v3`:

```text
snapshot SHA-256
f67c7bae50c4c279bf6372e098833be32699aca24232d7d489a1f7a45b5a8e21

historical source commit
9cf8e6922f35a5d6645e3128545953723bf54da2
```

| Arm | Two Wan calls | Two RF losses | Consistency weight | Role |
|---|---:|---:|---:|---|
| `TC-CONT` | yes | yes | 0.0 | two-forward matched control |
| `TC-CONS` | yes | yes | 0.2 | stopped-consistency candidate |

Weight `0.2` is fixed prospectively as a moderate auxiliary coefficient. It is
not tuned in this screen. Both arms compute `x0_hat_hi`, `x0_hat_lo`, the
normalized consistency diagnostic, and identical telemetry. The control does
not attach the zero-weight consistency graph to its returned objective.

The subclass adds no parameter or persistent buffer. Model parameter names,
shapes, count, actions, reference input, Wan I/O, and checkpoint topology are
identical to VPM. Each arm uses a fresh identical AdamW optimizer; this is a
matched model-only continuation, not an optimizer resume.

The forward-call budget and both expensive Wan forward/backward paths are
matched. The candidate additionally backpropagates through the inexpensive
elementwise consistency branch, whereas the control returns the exact base RF
loss and does not attach a zero-weight consistency graph. Consequently this
screen supports a same-update/same-Wan-call quality comparison, not a claim of
bit-exact total training FLOPs or equal wall-clock training time.

The inherited 64-channel VPM auxiliary topology remains a no-op in both arms:

- state and clock conditioning are off;
- auxiliary loss weight is zero;
- injection gates are exact zero;
- the unused auxiliary input is an all-zero tensor;
- no cached V-JEPA target is opened.

## Training controls and audit

Each arm runs exactly 200 updates with seed `1234`, eight B200 ranks, local
batch one, global batch eight, and the same immutable 512-row train manifest.
Both use AdamW (`lr=1e-4`, betas `0.9/0.95`), 20-step warmup, cosine decay to
`1e-6`, AMP, gradient clipping, no EMA, and the same checkpoint cadence.
Registration parses every manifest row and requires dense unique clip IDs,
unique source episodes, fixed 13-frame/stride-5 geometry, and zero train/val
overlap by clip ID, episode directory, or `(episode directory, start)`.
The trainer's rank-sharded validation iterator cycles because validation runs
at updates 0, 100, and 199; each event consumes exactly four local batches of
two, which is all 64 clips globally. The dedicated deployment evaluator below
still indexes the finite 64-clip split exactly once per endpoint.

Every update records rank-combined exact hashes of:

- clip IDs and actions;
- clean Wan latent and shared epsilon;
- high/low sigmas and corresponding native timesteps;
- high/low noisy states;
- CPU and CUDA RNG state after both calls.

It also records deterministic probes of the same tensors and both clock means.
Analysis fails before reading quality effects unless all 200 paired updates,
exact hashes, clip order, learning-rate values, and observation counts match.

Telemetry goes only to `zijiandu/dual-video-diffusion-private`, requires the
project to report `PRIVATE`, and uses `group=null`. Each W&B run ID is the
registration-bound arm identity with `resume=never`, so a fresh study root
cannot silently attach to an earlier failed or completed run.

## Deployment evaluation

The training subclass deliberately does not override any sampling method.
Evaluation uses the inherited ordinary VPM Euler sampler. For each of all 64
validation clips, the sampler receives only:

- five observed RGB frames;
- future requested action chunks and morphology;
- deterministic initial Gaussian noise.

It receives no held-out future RGB, clean future latent, feature target,
teacher output, or extra model call. Ground truth remains evaluator-owned until
all autonomous endpoints in a batch have completed.

Both arms are evaluated with paired input/noise hashes at total NFE
`{1,2,4}`. A Wan forward hook independently verifies that actual calls equal
declared NFE. NFE 1 is also run with actions shuffled within each fixed local
two-clip batch as a diagnostic; it cannot promote the candidate.

Metrics, lower is better:

- primary: decoded temporal-difference MSE in `[0,1]` units;
- guardrails: future video-latent NMSE and decoded future RGB MSE;
- diagnostics: future latent-delta NMSE, decoded PSNR, and action shuffle.

## Fixed decision gate

For each NFE and claim metric, paired relative improvement is

```text
I = [mean(TC-CONT) - mean(TC-CONS)] / mean(TC-CONT).
```

The analyzer uses 10,000 paired clip bootstraps with seed `20260807`. The
family has `3 NFE x 3 metrics = 9` predeclared contrasts. One-sided Bonferroni
lower bounds use confidence `1 - 0.05/9`.

An endpoint passes only when:

- decoded temporal MSE point estimate and simultaneous lower bound are both at
  least `+1%`; and
- latent NMSE and decoded MSE point estimates and simultaneous lower bounds are
  each greater than `-1%`.

The screen passes if any fixed endpoint passes, and reports the lowest passing
NFE. No checkpoint, loss weight, clock band, seed, NFE, or gate is changed
after registration. No protected test follows this screen.

## Interpretation

A positive result would show that an inference-free self-teacher can improve a
low-NFE video model without requiring a clean future feature. It would not be
"dual diffusion" at deployment: the second clock is training-only. It would
motivate multi-seed and longer-data confirmation, then distillation studies.

A null result would reject this exact stopped two-clock objective and weight
for the short VPM continuation. It would not prove all consistency training is
ineffective; teacher EMA, self-forced rollout consistency, or direct
one/two-step distillation are distinct hypotheses.

Training compute is intentionally about two Wan forwards per update in both
arms. The small consistency-loss backward overhead exists only in `TC-CONS`;
deployment compute and parameter count are unchanged.

## Implementation inventory

- `projects/latent_action_models/lam/two_clock_consistency_model.py`
- `projects/latent_action_models/train_two_clock_consistency.py`
- `robot_wm/utils/two_clock_consistency_trainer.py`
- `robot_wm/datasets/abc/two_clock_consistency_fixed_dataset.py`
- `tools/two_clock_consistency_evaluate.py`
- `tools/analyze_two_clock_consistency.py`
- `tools/slurm/two_clock_consistency_workflow.py`
- `tools/slurm/two_clock_consistency.sbatch`
- `docs/experiments/vpm_two_clock_consistency_runbook.md`

All artifacts are exclusive-create under a fresh Lustre root. Jobs are
non-requeueable and source registration requires an exact clean full commit.
