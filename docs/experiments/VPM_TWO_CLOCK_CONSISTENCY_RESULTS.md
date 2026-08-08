# Two-clock consistency screen results

Date completed: 2026-08-08

Study root:
`two-clock-consistency-seed1234-20260808-40d2e31-v2`

Jobs `505116` and `505117` completed with exit code `0:0`. The exact
preregistered analyzer produced SHA-256
`90592a9123575e61f036d1b8a3960fb091cf4240f0edb4789df0c17e134c95a0`.
No protected test was opened.

Positive changes favor `TC-CONS` over `TC-CONT`.

| NFE | Metric | CONT | CONS | Change | simultaneous LB |
|---:|---|---:|---:|---:|---:|
| 1 | latent NMSE | 0.24943570 | 0.24821420 | +0.4897% | -0.6243% |
| 1 | decoded MSE | 0.02015752 | 0.02002112 | +0.6767% | -0.6691% |
| 1 | temporal MSE | 0.01336272 | 0.01339966 | -0.2764% | -0.6586% |
| 2 | latent NMSE | 0.24956772 | 0.24897049 | +0.2393% | -0.8640% |
| 2 | decoded MSE | 0.02014552 | 0.02002943 | +0.5763% | -0.8028% |
| 2 | temporal MSE | 0.01336255 | 0.01342076 | -0.4356% | -0.8066% |
| 4 | latent NMSE | 0.30140662 | 0.30307473 | -0.5534% | -1.9662% |
| 4 | decoded MSE | 0.02419116 | 0.02428758 | -0.3986% | -2.0199% |
| 4 | temporal MSE | 0.01560174 | 0.01594730 | -2.2149% | -3.3182% |

No NFE passed. The consistency objective trades small low-NFE appearance
improvements for worse temporal coherence, and by NFE 4 it harms all three
metrics. Aligned-versus-shuffled action effects remained below 0.05% in
magnitude.

Conclusion: this narrow stopped two-clock consistency objective is not a
substitute for an inference-time predictive motion state. It is also not a
full implementation or test of Causal-rCM-style distillation.

