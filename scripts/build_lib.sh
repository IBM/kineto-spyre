#!/bin/bash
# ---------------------------------------------------------------------------
# build_lib.sh — guard functions for the kineto-spyre 2.12 release build.
#
# This is a *sourced* helper: scripts/build_pytorch.sh sources it and calls the
# functions below as gates around the real (multi-GB, AIU-hardware-dependent)
# PyTorch build. Each function is small and independently callable so the gate
# logic can be unit-tested in isolation WITHOUT cloning PyTorch or running a
# real build (see scripts/test_build_guards.py).
#
# Every gate "fails closed": on any error it writes a clear message to stderr
# (and, where relevant, to build.log) and returns a non-zero status so the
# caller can stop before producing a downstream artifact (a wheel).
#
# Design reference: design.md Component 4 (Build Script), sections 4a–4f.
# Requirements: 2.3, 2.7, 3.1, 3.2, 4.1–4.6, 5.1–5.8, 9.1–9.5.
#
# The functions can also be driven directly for testing:
#     bash scripts/build_lib.sh <function_name> [args...]
# ---------------------------------------------------------------------------

# Exit status conventions (used by callers and tests).
EXIT_VERSION_MISMATCH=1     # Req 2.7 / 5.7: source not 2.12
EXIT_REPLACE_FAILED=1       # Req 5.8: kineto replacement incomplete
EXIT_PRIVATEUSE1_MISSING=1  # Req 3.2: PrivateUse1_Registration absent
EXIT_AIUPTI_MISSING=3       # Req 4.4: libaiupti not detected (hard gate)
EXIT_SUBCOMPONENT=1         # Req 9.3/9.4/9.5: subcomponent version problem
EXIT_WHEEL_COUNT=1          # Req 5.4/5.6: not exactly one wheel

# The line the build script appends to build.log describing AIUPTI detection.
AIUPTI_DETECTED_PREFIX="AIUPTI detected:"
AIUPTI_NOT_DETECTED_MSG="AIUPTI not detected: libaiupti was not found"

# ---------------------------------------------------------------------------
# Small JSON helpers (python3 based so we do not depend on jq being installed).
# ---------------------------------------------------------------------------

# _json_subcomponent_keys <release_record.json> -> prints one key per line
_json_subcomponent_keys() {
  local record="$1"
  python3 - "$record" <<'PY'
import json, sys
with open(sys.argv[1]) as fh:
    data = json.load(fh)
for key in data.get("subcomponents", {}):
    print(key)
PY
}

# _json_subcomponent_value <release_record.json> <key> -> prints the value
# (prints nothing for a missing/empty value). Returns non-zero only on a JSON
# read error, so callers distinguish "key missing/empty" (printed empty) from
# "record unreadable".
_json_subcomponent_value() {
  local record="$1" key="$2"
  python3 - "$record" "$key" <<'PY'
import json, sys
with open(sys.argv[1]) as fh:
    data = json.load(fh)
val = data.get("subcomponents", {}).get(sys.argv[2])
if val is None:
    print("")
else:
    print(val)
PY
}

# ---------------------------------------------------------------------------
# 4a. Version pin / source guard (Req 2.7, 5.7).
# ---------------------------------------------------------------------------

# verify_pytorch_version <path/to/version.txt>
# Stops (non-zero) and reports a mismatch unless the source version is 2.12.*.
verify_pytorch_version() {
  local version_file="$1"
  if [ ! -f "$version_file" ]; then
    echo "ERROR: PyTorch version file not found: $version_file" >&2
    return "$EXIT_VERSION_MISMATCH"
  fi
  local actual
  actual="$(tr -d '[:space:]' < "$version_file")"
  case "$actual" in
    2.12.*)
      echo "PyTorch source version verified: $actual"
      return 0
      ;;
    *)
      echo "ERROR: PyTorch source is '$actual', expected 2.12.x — stopping before build" >&2
      return "$EXIT_VERSION_MISMATCH"
      ;;
  esac
}

# assert_build_version <build_version_string>
# Smoke check (Req 2.3, 5.2): the build version must identify 2.12.
assert_build_version() {
  local build_version="$1"
  case "$build_version" in
    2.12.*)
      echo "Build version verified: $build_version"
      return 0
      ;;
    *)
      echo "ERROR: build version '$build_version' does not target 2.12" >&2
      return "$EXIT_VERSION_MISMATCH"
      ;;
  esac
}

# ---------------------------------------------------------------------------
# 4b. Kineto replacement guard (Req 5.3, 5.8).
# ---------------------------------------------------------------------------

# verify_kineto_replacement <pytorch_root>
# Confirms the fork replaced third_party/kineto: the AIU plugin dir must be
# present (fork marker). Stops + reports a replacement failure otherwise.
verify_kineto_replacement() {
  local pytorch_root="$1"
  local plugin_dir="$pytorch_root/third_party/kineto/libkineto/src/plugin/aiupti"
  if [ ! -d "$plugin_dir" ]; then
    echo "ERROR: kineto replacement failed — AIU plugin dir missing: $plugin_dir" >&2
    return "$EXIT_REPLACE_FAILED"
  fi
  echo "kineto replacement verified: AIU plugin present at $plugin_dir"
  return 0
}

# ---------------------------------------------------------------------------
# 4c. PrivateUse1 registration source check (Req 3.1, 3.2).
# ---------------------------------------------------------------------------

# check_privateuse1 <pytorch_root> <build_log>
# Greps the obtained PyTorch source for the PrivateUse1 registration macro/API.
# If absent: halts BEFORE the wheel build (returns non-zero so no wheel is
# produced) and emits a build.log entry naming PrivateUse1_Registration as the
# missing dependency (Req 3.2).
check_privateuse1() {
  local pytorch_root="$1" build_log="$2"
  local search_root="$pytorch_root/torch"
  [ -d "$search_root" ] || search_root="$pytorch_root"
  if grep -rqs "REGISTER_PRIVATEUSE1_PROFILER\|registerPrivateUse1Activity" "$search_root"; then
    echo "PrivateUse1_Registration present in PyTorch source"
    return 0
  fi
  local msg="ERROR: missing dependency PrivateUse1_Registration (REGISTER_PRIVATEUSE1_PROFILER / registerPrivateUse1Activity) — halting before wheel build, no wheel produced"
  echo "$msg" >&2
  if [ -n "$build_log" ]; then
    echo "$msg" >> "$build_log"
  fi
  return "$EXIT_PRIVATEUSE1_MISSING"
}

# ---------------------------------------------------------------------------
# 4d. AIUPTI detection hard gate + build-log assertion (Req 4.1–4.6).
# ---------------------------------------------------------------------------

# assert_aiupti_detected <build_log>
# Scans build.log for CMake's "AIU library found: <path>" status line. Emits a
# build.log entry stating libaiupti detected (with the resolved path) or not
# detected (Req 4.3). On detection: returns 0. When NOT detected: aborts with a
# non-zero status (Req 4.4) so the caller produces no wheel.
#
# This gate is UNCONDITIONAL (Req 4.6): it fires regardless of build type, with
# no override flag and no warning-and-continue path. There is intentionally no
# parameter or environment variable that downgrades a missing detection to a
# warning.
assert_aiupti_detected() {
  local build_log="$1"
  if [ -z "$build_log" ] || [ ! -f "$build_log" ]; then
    echo "ERROR: build log not found for AIUPTI detection: $build_log" >&2
    return "$EXIT_AIUPTI_MISSING"
  fi
  local found_line
  found_line="$(grep 'AIU library found:' "$build_log" | tail -1)"
  if [ -n "$found_line" ]; then
    local resolved_path
    resolved_path="$(printf '%s' "$found_line" | sed -e 's/.*AIU library found:[[:space:]]*//')"
    local log_line="$AIUPTI_DETECTED_PREFIX $resolved_path"
    echo "$log_line"
    echo "$log_line" >> "$build_log"
    return 0
  fi
  echo "$AIUPTI_NOT_DETECTED_MSG" >> "$build_log"
  echo "ERROR: libaiupti NOT detected — aborting release build (a wheel would emit no AIU events)" >&2
  return "$EXIT_AIUPTI_MISSING"
}

# confirm_wheel_has_aiupti <build_log>
# Post-build confirmation (Req 4.5) that the wheel was built with AIUPTI: the
# detection line must be present in build.log. Returns non-zero otherwise.
confirm_wheel_has_aiupti() {
  local build_log="$1"
  if [ -n "$build_log" ] && [ -f "$build_log" ] \
    && grep -q "^$AIUPTI_DETECTED_PREFIX" "$build_log"; then
    echo "Confirmed: wheel built with HAS_AIUPTI"
    return 0
  fi
  echo "ERROR: could not confirm the wheel was built with HAS_AIUPTI" >&2
  return "$EXIT_AIUPTI_MISSING"
}

# ---------------------------------------------------------------------------
# 4e. Subcomponent version verification (Req 9.2–9.5).
# ---------------------------------------------------------------------------

# resolve_subcomponent_version <subcomponent>
# Returns the *obtained* version of a subcomponent on stdout, or non-zero when
# the version cannot be obtained (Req 9.4). For unit tests the obtained
# versions are supplied via the OBTAINED_VERSIONS_JSON file (a JSON object
# mapping subcomponent -> obtained version); a missing key means "unobtainable".
# In a real build this is where pytorch/version.txt, the libaiupti install, the
# AIU toolkit, and the fork version would be probed.
resolve_subcomponent_version() {
  local sub="$1"
  if [ -n "${OBTAINED_VERSIONS_JSON:-}" ]; then
    local v
    v="$(_json_subcomponent_value "$OBTAINED_VERSIONS_JSON" "$sub")" || return 1
    [ -n "$v" ] || return 1
    printf '%s\n' "$v"
    return 0
  fi
  # Real resolution (executed on a build host).
  case "$sub" in
    pytorch)
      tr -d '[:space:]' < "${PYTORCH_ROOT:-pytorch}/version.txt" 2>/dev/null || return 1
      ;;
    *)
      # Other subcomponents (libaiupti, aiu_toolkit, kineto_spyre) are resolved
      # from the install/toolkit on the build host; without an override we
      # cannot obtain them here.
      return 1
      ;;
  esac
}

# verify_subcomponents <release_record.json>
# For each subcomponent in the record: build against the recorded version,
# reporting-and-stopping on a missing recorded version (Req 9.3), an
# unobtainable version (Req 9.4), or a recorded-vs-obtained mismatch naming the
# subcomponent and both versions (Req 9.5).
verify_subcomponents() {
  local record="$1"
  if [ ! -f "$record" ]; then
    echo "ERROR: release record not found: $record" >&2
    return "$EXIT_SUBCOMPONENT"
  fi
  local keys
  keys="$(_json_subcomponent_keys "$record")" || {
    echo "ERROR: could not read subcomponents from $record" >&2
    return "$EXIT_SUBCOMPONENT"
  }
  if [ -z "$keys" ]; then
    echo "ERROR: release record has no subcomponents: $record" >&2
    return "$EXIT_SUBCOMPONENT"
  fi
  local sub want got
  while IFS= read -r sub; do
    [ -n "$sub" ] || continue
    want="$(_json_subcomponent_value "$record" "$sub")"
    if [ -z "$want" ]; then
      echo "ERROR: no recorded version for subcomponent '$sub' — stopping before build" >&2
      return "$EXIT_SUBCOMPONENT"
    fi
    if ! got="$(resolve_subcomponent_version "$sub")" || [ -z "$got" ]; then
      echo "ERROR: subcomponent '$sub' version '$want' is recorded but cannot be obtained — stopping before build" >&2
      return "$EXIT_SUBCOMPONENT"
    fi
    if [ "$want" != "$got" ]; then
      echo "ERROR: subcomponent '$sub' version mismatch: recorded '$want', obtained '$got' — stopping before build" >&2
      return "$EXIT_SUBCOMPONENT"
    fi
    echo "subcomponent '$sub' verified at version '$want'"
  done <<EOF
$keys
EOF
  return 0
}

# ---------------------------------------------------------------------------
# 4f. Wheel production helpers (Req 5.4, 5.6).
# ---------------------------------------------------------------------------

# clean_dist <dist_dir>
# Clears dist/ so no stale/partial wheel can be mistaken for the build output.
clean_dist() {
  local dist_dir="$1"
  rm -rf "$dist_dir"
  mkdir -p "$dist_dir"
  echo "cleared $dist_dir"
}

# assert_single_wheel <dist_dir>
# Postcondition (Req 5.4): exactly one wheel in dist/. On a count of 0 or 2+
# (Req 5.6) reports the failure, removes any partial wheel, and returns
# non-zero so nothing is published.
assert_single_wheel() {
  local dist_dir="$1"
  local count
  count="$(find "$dist_dir" -maxdepth 1 -name '*.whl' 2>/dev/null | wc -l | tr -d '[:space:]')"
  if [ "$count" -eq 1 ]; then
    echo "single-wheel postcondition satisfied: $(find "$dist_dir" -maxdepth 1 -name '*.whl')"
    return 0
  fi
  echo "ERROR: expected exactly 1 wheel in $dist_dir, found $count — publishing nothing, removing partial wheel(s)" >&2
  rm -f "$dist_dir"/*.whl 2>/dev/null || true
  return "$EXIT_WHEEL_COUNT"
}

# report_build_failure <stage>
# Reports the failing build stage, leaves no partial wheel (Req 5.6).
report_build_failure() {
  local stage="$1" dist_dir="${2:-dist}"
  echo "ERROR: build failed at stage: $stage — publishing nothing, cleaning partial wheel(s)" >&2
  rm -f "$dist_dir"/*.whl 2>/dev/null || true
  return 1
}

# ---------------------------------------------------------------------------
# Direct-invocation dispatcher (for tests / manual --check use).
# When sourced, BASH_SOURCE[0] != $0 and this block is skipped.
# ---------------------------------------------------------------------------
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  _fn="$1"
  shift || true
  if [ -z "$_fn" ]; then
    echo "usage: $0 <function> [args...]" >&2
    exit 64
  fi
  "$_fn" "$@"
  exit $?
fi
