# Observed-anchor V-JEPA increment screen results

Date completed: 2026-08-08

Study root:
`observed-anchor-seed1234-20260808-23ab08c-v3`

Jobs `505106`, `505107`, and `505108` completed with exit code `0:0`. The
development decision is `frozen_no_selection`; zero protected-test examples
or caches were opened. Both arms completed 5,000 updates with 41,963,760
parameters and matched source/data/provenance.

| NFE | C-ABS semantic NMSE | AINC semantic NMSE | AINC cosine | AINC temporal NMSE |
|---:|---:|---:|---:|---:|
| 1 | 0.615067 | 0.315475 | 0.844884 | 0.999274 |
| 2 | 0.671990 | 0.340561 | 0.834117 | 1.039335 |
| 4 | 0.767122 | 0.414095 | 0.807499 | 1.170097 |

AINC improves semantic NMSE over C-ABS by 48.709%, 49.320%, and 46.020%.
That gain is real but comes from the clean observed-prefix anchor preserving
static scene/object identity:

- autonomous increment NMSE is 1.0012, 1.0434, and 1.1736;
- increment cosine is 0.0495, 0.0257, and 0.0181;
- a train-mean increment has temporal NMSE 0.997082 and is better overall;
- the static-anchor temporal NMSE of 1.0 is better at NFE 2 and 4;
- temporal improvements versus context-shuffled and donor-target controls are
  below 0.3%, far short of the required 5% attribution threshold.

Conclusion: an inference-available V-JEPA history anchor is useful for static
identity but does not yield a predictive future-motion state. More denoising
steps worsen the generated increments. The screen therefore cannot support a
video-quality or dual-diffusion claim.

