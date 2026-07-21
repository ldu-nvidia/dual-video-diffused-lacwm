# Working agreements

- Treat the production LACWM checkpoint and run directories as immutable.
- Never launch, stop, requeue, or modify a cluster job without explicit scope.
- Put checkpoints, datasets, logs, and generated videos outside this Git repo on
  approved Lustre, `/mnt/data1`, or `/mnt/data2` storage.
- Keep `dual_diffusion.enabled=false` until the current phase gate passes.
- Use the explicit-action model for causal forecasting and DAgger evaluation;
  the latent-action inverse model may consume future RGB.
- Compute transforms independently per camera view.  Never apply an FFT across
  the artificial width boundary between stacked views.
- State the clock convention in every schedule: LACWM uses `sigma=1` for noise
  and `sigma=0` for clean data.
- Ground quality and speed claims in saved configs, checkpoints, and evaluation
  artifacts.  Label proposed mechanisms as hypotheses until measured.
- Add endpoint, sign, mask, view-isolation, and baseline-no-op tests with every
  schedule or architecture change.
