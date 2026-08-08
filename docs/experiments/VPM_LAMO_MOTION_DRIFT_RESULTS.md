# LaMo macro-motion drift screen results

Date completed: 2026-08-08

Study root:
`lamo-motion-drift-seed1234-20260808-40d2e31-v2`

Jobs `505114` and `505115` completed with exit code `0:0`. The exact
preregistered analyzer produced SHA-256
`338c07dfe3e770cff037c99dae3adcb9090df71b0c3312d0726a1e8b3e1e56c8`.
No protected test was opened.

Positive changes favor `VPM-DRIFT` over the matched `VPM-CONT` continuation.
`LB` is the preregistered 99.444% one-sided simultaneous lower bound.

| NFE | Metric | CONT | DRIFT | Change | LB |
|---:|---|---:|---:|---:|---:|
| 1 | latent NMSE | 0.24024001 | 0.24008233 | +0.0656% | -1.0280% |
| 1 | decoded MSE | 0.01926273 | 0.01929985 | -0.1927% | -1.2505% |
| 1 | temporal MSE | 0.01334251 | 0.01335227 | -0.0731% | -0.3780% |
| 2 | latent NMSE | 0.24041827 | 0.24029612 | +0.0508% | -1.0841% |
| 2 | decoded MSE | 0.01925084 | 0.01929100 | -0.2086% | -1.2354% |
| 2 | temporal MSE | 0.01334593 | 0.01336124 | -0.1147% | -0.4217% |
| 4 | latent NMSE | 0.29280366 | 0.29215937 | +0.2200% | -1.2320% |
| 4 | decoded MSE | 0.02361059 | 0.02364361 | -0.1399% | -1.5764% |
| 4 | temporal MSE | 0.01582801 | 0.01575380 | +0.4688% | -0.6943% |

No NFE passed. The largest temporal point effect was only +0.469% at NFE 4,
below the required +1%, and its simultaneous bound was negative. The NFE-1
aligned-versus-action-shuffled effects were between -0.066% and +0.037%, so
the screen also found no sample-specific action attribution.

Conclusion: the parameter-free training-only macro-motion loss is effectively
neutral at this 200-update, one-seed budget. It does not create a deployable
motion prior or improve the one-call frontier.

