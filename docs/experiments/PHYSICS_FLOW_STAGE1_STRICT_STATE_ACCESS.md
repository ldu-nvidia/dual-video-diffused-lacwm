# Raw physics-flow strict state access repair

Status: source-only repair; no cache, registration, cluster job, W&B run, or
experiment outcome was created or opened by this branch.

## Problem

The previous cache builder used `np.load(states.npz)[member][slice]`.  Although
the Python expression selected only frame 4, NumPy's NPZ path first opened and
materialized the complete member.  Consequently, the receipt's claim that
future measured state was not opened was stronger than the physical access
boundary actually enforced.

## Enforced storage and access contract

Cache schema `raw-physics-flow-cache-v3` now requires the canonical six-member
ABC archive, in order:

1. `joint_states.npy`
2. `joint_actions.npy`
3. `gripper_states.npy`
4. `gripper_actions.npy`
5. `frame_ts.npy`
6. `instruction.npy`

Every member must be `ZIP_STORED`, unencrypted, have no data descriptor, and
have agreeing local and central metadata.  The four accessed NPY arrays must
be little-endian float32, two-dimensional, C-contiguous, mutually equal in
frame count, and have widths 12, 12, 2, and 2.  NPY and ZIP member lengths must
agree exactly, and archive comments, extra members, missing members, trailing
bytes, truncated bytes, and noncanonical ZIP metadata fail closed.

DEFLATE is deliberately unsupported.  A decompressor may consume compressed
bytes that encode values beyond the requested uncompressed element boundary.
Such an archive is rejected from its local header before any member payload is
read.  A compressed source would first require a prospectively generated,
observed-only `ZIP_STORED`/mmap-sliceable sidecar.

For a valid archive, bounded NPY headers are parsed first.  The builder then
uses exact `os.pread` calls for only:

- `joint_states[frame4:frame4+1]`;
- `gripper_states[frame4:frame4+1]`;
- `joint_actions[start:start+65]`; and
- `gripper_actions[start:start+65]`.

The per-row ledger records every selected archive byte interval and hash.  A
runtime invariant rejects any read intersecting unselected NPY array data.
For the registered geometry, 3,696 selected array-data bytes are returned:
`(14 + 65 * 14) * 4`.  Future measured-state, unregistered-action, and unused
member-payload array-data counts must each be exactly zero.

ZIP directory/local-header metadata necessarily contains whole-member CRC32
fields.  The reader requires the two stored CRC metadata values to agree as a
structural check, but deliberately does not return the value or validate it by
reading the complete member.  It does not claim zero physical archive reads
after a selected element; it claims—and audits—zero unselected **NPY
array-data** bytes returned to userspace.  Whole-file mtime, inode, CRC value,
and digest are excluded from the selected-value receipt because each can
summarize or change with future values.

## Prospective preflight

Registration validates every train and validation descriptor before the cache
output root is created.  Each row receives an identity-bound structural
receipt and exact planned direct-read ranges; preflight returns zero array-data
bytes.  `validate_cache_registration` independently reconstructs the complete
train-plus-validation inventory before either split can create a cache root.
The row reader repeats all structural checks when it performs the selected
reads.

## Value equivalence and tests

For a conforming float32 C-order archive, direct byte decoding is value- and
bit-equivalent to selecting the same rows from the former NumPy-loaded arrays;
the endpoint selection and renderer inputs are unchanged.  This is established
on deterministic synthetic archives, not claimed from an experiment artifact.

Adversarial tests establish that:

- mutations to both measured-state arrays after frame 4 leave every extracted
  tensor and selected-value receipt identical;
- a frame-4 mutation changes the extracted state and its receipt hash;
- a compressed archive is rejected before any NPY/member payload read;
- every registered descriptor receives a zero-array-data preflight ledger; and
- wrong dtype, shape/order, member order/name, compression, extra/trailing
  content, and premature/truncated content fail closed.

This schema is intentionally incompatible with v2 caches.  Adoption requires
a fresh source stage, registration, and train/validation cache namespace.
