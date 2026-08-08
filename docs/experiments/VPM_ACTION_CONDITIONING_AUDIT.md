# VPM explicit-action conditioning audit

Date completed: 2026-08-08

This read-only diagnostic exactly replayed the trained action encoder,
four-to-one temporal pool, and Wan `ActionToControl` projection from the frozen
historical VPM update-1,000 snapshot over all 512 registered training action
clips. It read no RGB, future target, validation, or protected-test data.

Artifact:
`action_conditioning_audit/vpm-action-conditioning-df3dbb7/action_conditioning.json`

- artifact SHA-256:
  `da194367f77259067bb5e81838a81118a107a2473c3c193cbbe412e7380c131e`;
- analysis identity:
  `9d9844c8ebada728094a794902f72d7c48562961228a60677377edf3c595f067`;
- snapshot SHA-256:
  `f67c7bae50c4c279bf6372e098833be32699aca24232d7d489a1f7a45b5a8e21`;
- action-array SHA-256:
  `f2cde809c1d864d4a00422aca8fcac0116229a0b0ac83a93850d1421d16c5b89`.

## Result

| State | RMS | sample std / RMS | cyclic-shuffle difference / RMS | cyclic-shuffle cosine |
|---|---:|---:|---:|---:|
| padded future actions | 0.25644 | 0.53611 | 0.75454 | 0.72805 |
| encoded action | 0.37831 | 0.17152 | 0.23758 | 0.99806 |
| pooled action | 0.60197 | 0.15951 | 0.22090 | 0.99883 |
| Wan control | 0.35152 | 0.16994 | 0.23539 | 0.99885 |

The centered 32-value Wan control has effective rank only `1.164`. Although
the action-dependent control component has substantial absolute RMS (`0.2969`,
84.46% of total control RMS), it points in almost the same direction for most
clips: correct-versus-zero control cosine is `0.97597`, while cyclically
shuffled controls have cosine `0.99885`.

The raw requested action sequences are much more sample-specific
(shuffle cosine `0.728`, relative shuffle difference `0.755`). Most of that
identity is lost before Wan. In combination with the repeated sub-0.3% video
effects of action shuffling, this supports a concrete bottleneck hypothesis:
the current path mostly signals a common "robot is acting" direction rather
than a rich clip-specific motion request.

This is an observational representation audit, not proof that the low-rank
control causes all video failures. The natural controlled follow-up is an
initial-function-preserving, train-statistics-fitted action whitening/delta
adapter with aligned, shuffled, and zero-action video endpoints.

