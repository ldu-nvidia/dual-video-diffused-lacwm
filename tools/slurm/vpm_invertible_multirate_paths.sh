#!/usr/bin/env bash

# Source-only path guard for the frozen ILSF-2 launcher. The caller supplies a
# canonical artifact prefix whose parent already exists. This function may
# create exactly the one experiment-family directory; it never creates the run
# output itself and never removes or overwrites an existing path.
prepare_vpm_invertible_multirate_artifact_parent() {
  local output_dir="${1:?output directory is required}"
  local runtime_record="${2:?runtime record is required}"
  local artifact_prefix="${3:?artifact prefix is required}"
  local expected_parent="${artifact_prefix%/}"
  local observed_parent grandparent canonical_grandparent canonical_parent

  [[ "$output_dir" == /* && "$runtime_record" == /* ]] || return 20
  [[ "$artifact_prefix" == /*/ ]] || return 21
  [[ "$output_dir" == "$artifact_prefix"* ]] || return 22
  [[ "$runtime_record" == "${output_dir}.runtime_verification.json" ]] || return 23
  observed_parent="$(dirname -- "$output_dir")"
  [[ "$observed_parent" == "$expected_parent" ]] || return 24
  [[ ! -e "$output_dir" && ! -L "$output_dir" ]] || return 25
  [[ ! -e "$runtime_record" && ! -L "$runtime_record" ]] || return 26

  grandparent="$(dirname -- "$expected_parent")"
  [[ -d "$grandparent" && ! -L "$grandparent" ]] || return 27
  canonical_grandparent="$(cd "$grandparent" && pwd -P)" || return 28
  [[ "$expected_parent" == "$canonical_grandparent/$(basename -- "$expected_parent")" ]] || return 29

  if [[ -e "$expected_parent" || -L "$expected_parent" ]]; then
    [[ -d "$expected_parent" && ! -L "$expected_parent" ]] || return 30
  else
    mkdir -- "$expected_parent" || return 31
  fi
  canonical_parent="$(cd "$expected_parent" && pwd -P)" || return 32
  [[ "$canonical_parent" == "$expected_parent" ]] || return 33
  [[ ! -e "$output_dir" && ! -L "$output_dir" ]] || return 34
  [[ ! -e "$runtime_record" && ! -L "$runtime_record" ]] || return 35
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "ERROR: source this helper from the ILSF-2 launcher" >&2
  exit 2
fi
