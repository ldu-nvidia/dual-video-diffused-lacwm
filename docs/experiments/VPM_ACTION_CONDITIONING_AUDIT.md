# VPM explicit-action conditioning audit

Date completed: 2026-08-08

This read-only diagnostic exactly replayed the trained action encoder,
four-to-one temporal pool, and Wan `ActionToControl` projection from the frozen
historical VPM update-1,000 snapshot over all 512 registered training action
clips. It read no RGB, future target, validation, or protected-test data.

Artifact:
`action_conditioning_audit/vpm-action-conditioning-97ab511-v2/action_conditioning.json`

- artifact SHA-256:
  `7c3b4d6855450b3d0290161a41410e881c19131c76c7de6bbbce0112fe383403`;
- analysis identity:
  `af3c91f285037c047214f900727b1e47af5071cda775ff3ecca53780a336f98e`;
- snapshot SHA-256:
  `f67c7bae50c4c279bf6372e098833be32699aca24232d7d489a1f7a45b5a8e21`;
- action-array SHA-256:
  `f2cde809c1d864d4a00422aca8fcac0116229a0b0ac83a93850d1421d16c5b89`.

## Result

| State | effective rank | sample std / RMS | cyclic-shuffle difference / RMS | cyclic-shuffle cosine |
|---|---:|---:|---:|---:|
| padded future actions | 14.005 | 0.53611 | 0.75454 | 0.72805 |
| within-chunk action delta | 92.128 | 0.99954 | 1.41039 | 0.00663 |
| chunk endpoint delta | 53.868 | 0.99941 | 1.40943 | 0.00808 |
| encoded action | 1.336 | 0.17152 | 0.23758 | 0.99806 |
| pooled action | 1.168 | 0.15951 | 0.22090 | 0.99883 |
| Wan control | 1.164 | 0.16994 | 0.23539 | 0.99885 |

The centered 32-value Wan control has effective rank only `1.164`. Although
the action-dependent control component has substantial absolute RMS (`0.2969`,
84.46% of total control RMS), it points in almost the same direction for most
clips: correct-versus-zero control cosine is `0.97597`, while cyclically
shuffled controls have cosine `0.99885`.

The raw requested action sequences are much more sample-specific
(shuffle cosine `0.728`, effective rank `14.0`). Their within-chunk temporal
deltas are richer still (shuffle cosine `0.0066`, effective rank `92.1`). Most
of that identity is lost in the first learned action embedding before Wan. In
combination with the repeated sub-0.3% video effects of action shuffling, this
supports a concrete bottleneck hypothesis: the current path mostly signals a
common "robot is acting" direction rather than a rich clip-specific motion
request.

This is an observational representation audit, not proof that the low-rank
control causes all video failures. The natural controlled follow-up is an
initial-function-preserving, train-statistics-fitted action-delta residual
adapter with aligned, shuffled, and zero-action video endpoints.
