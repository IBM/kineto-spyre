#!/usr/bin/env bash
#
# record_sync.sh — Release Record + Provenance (design Component 3)
#
# Writes/updates release_record.json with the integrated upstream commit IDs
# and the new Last_Sync_Commit, and rewrites the README sync line to point at
# the newly synced upstream commit while reflecting the PyTorch 2.12 target.
# The produced record is then validated with scripts/validate_release_record.py.
#
# Requirements: 1.2 (record every integrated SHA), 1.3 (record newest as the new
# Last_Sync_Commit), 1.5 (update the README sync line).
#
# Every input is parameterised (flags or environment) so the script can be
# exercised against fixture repositories without touching live repos.
#
# Usage:
#   record_sync.sh \
#       --repo PATH \
#       --last-sync <40-hex> \
#       --pinned-kineto-commit <40-hex> \
#       [--target-pytorch 2.12.0] \
#       [--release "torch-2.12.0+aiu.kineto.1.2.0"] \
#       --libaiupti-version X --aiu-toolkit-version X \
#       [--pytorch-version 2.12.0] --kineto-spyre-version X
#
# Each flag also has an environment-variable equivalent (REPO, LAST_SYNC,
# PINNED_KINETO_COMMIT, TARGET_PYTORCH, RELEASE, LIBAIUPTI_VERSION,
# AIU_TOOLKIT_VERSION, PYTORCH_VERSION, KINETO_SPYRE_VERSION, RECORD, README,
# VALIDATOR).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---- Defaults (overridable by env, then by flags) -------------------------
REPO="${REPO:-$DEFAULT_REPO}"
LAST_SYNC="${LAST_SYNC:-}"
PINNED_KINETO_COMMIT="${PINNED_KINETO_COMMIT:-}"
TARGET_PYTORCH="${TARGET_PYTORCH:-2.12.0}"
LIBAIUPTI_VERSION="${LIBAIUPTI_VERSION:-}"
AIU_TOOLKIT_VERSION="${AIU_TOOLKIT_VERSION:-}"
PYTORCH_VERSION="${PYTORCH_VERSION:-}"
KINETO_SPYRE_VERSION="${KINETO_SPYRE_VERSION:-}"
RELEASE="${RELEASE:-}"
RECORD="${RECORD:-}"
README="${README:-}"
VALIDATOR="${VALIDATOR:-${SCRIPT_DIR}/validate_release_record.py}"

usage() {
  sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

die() { echo "ERROR: $*" >&2; exit 1; }

# ---- Flag parsing ----------------------------------------------------------
while [ "$#" -gt 0 ]; do
  case "$1" in
    --repo)                  REPO="$2"; shift 2;;
    --last-sync)             LAST_SYNC="$2"; shift 2;;
    --pinned-kineto-commit)  PINNED_KINETO_COMMIT="$2"; shift 2;;
    --target-pytorch)        TARGET_PYTORCH="$2"; shift 2;;
    --release)               RELEASE="$2"; shift 2;;
    --libaiupti-version)     LIBAIUPTI_VERSION="$2"; shift 2;;
    --aiu-toolkit-version)   AIU_TOOLKIT_VERSION="$2"; shift 2;;
    --pytorch-version)       PYTORCH_VERSION="$2"; shift 2;;
    --kineto-spyre-version)  KINETO_SPYRE_VERSION="$2"; shift 2;;
    --record)                RECORD="$2"; shift 2;;
    --readme)                README="$2"; shift 2;;
    --validator)             VALIDATOR="$2"; shift 2;;
    -h|--help)               usage; exit 0;;
    *) die "unknown argument: $1";;
  esac
done

# ---- Derive remaining defaults --------------------------------------------
PYTORCH_VERSION="${PYTORCH_VERSION:-$TARGET_PYTORCH}"
RECORD="${RECORD:-${REPO}/release_record.json}"
README="${README:-${REPO}/README.md}"
RELEASE="${RELEASE:-torch-${TARGET_PYTORCH}+aiu.kineto.${KINETO_SPYRE_VERSION}}"

# ---- Validate inputs -------------------------------------------------------
[ -d "$REPO" ] || die "repo path does not exist: $REPO"
[ -n "$LAST_SYNC" ] || die "--last-sync (LAST_SYNC) is required"
[ -n "$PINNED_KINETO_COMMIT" ] || die "--pinned-kineto-commit (PINNED_KINETO_COMMIT) is required"
echo "$LAST_SYNC" | grep -Eq '^[0-9a-f]{40}$' || die "LAST_SYNC is not a 40-char SHA: $LAST_SYNC"
echo "$PINNED_KINETO_COMMIT" | grep -Eq '^[0-9a-f]{40}$' || die "PINNED_KINETO_COMMIT is not a 40-char SHA: $PINNED_KINETO_COMMIT"
[ -n "$LIBAIUPTI_VERSION" ]    || die "--libaiupti-version is required (Req 9.1)"
[ -n "$AIU_TOOLKIT_VERSION" ]  || die "--aiu-toolkit-version is required (Req 9.1)"
[ -n "$PYTORCH_VERSION" ]      || die "--pytorch-version is required (Req 9.1)"
[ -n "$KINETO_SPYRE_VERSION" ] || die "--kineto-spyre-version is required (Req 9.1)"
[ -f "$README" ] || die "README not found: $README"
[ -f "$VALIDATOR" ] || die "validator not found: $VALIDATOR"

# ---- 1. Extract integrated upstream SHAs in ascending order (Req 1.2) ------
# Each cherry-picked commit carries an `-x` provenance line of the form
# "(cherry picked from commit <40-hex>)". Reading them with --reverse yields
# the integrated commits in ascending chronological order.
INTEGRATED="$(
  git -C "$REPO" log --reverse --pretty=%B "${LAST_SYNC}..HEAD" \
    | grep -oE 'cherry picked from commit [0-9a-f]{40}' \
    | awk '{print $5}' || true
)"

# ---- 2. Determine the new Last_Sync_Commit (Req 1.3) -----------------------
# The newest integrated commit is the last one in ascending order. By the
# cherry-pick range (… last inclusive) it must equal the pinned commit; the
# schema validator enforces new_last_sync_commit == pinned_kineto_commit.
if [ -n "$INTEGRATED" ]; then
  NEW_LAST_SYNC="$(printf '%s\n' "$INTEGRATED" | tail -n 1)"
else
  NEW_LAST_SYNC="$PINNED_KINETO_COMMIT"
fi

if [ "$NEW_LAST_SYNC" != "$PINNED_KINETO_COMMIT" ]; then
  echo "WARNING: newest integrated commit ($NEW_LAST_SYNC) != pinned commit ($PINNED_KINETO_COMMIT);" >&2
  echo "         the release record will fail schema validation (invariant new==pinned)." >&2
fi

# ---- 3. Write release_record.json -----------------------------------------
INTEGRATED_FILE="$(mktemp)"
TMP_README="$(mktemp)"
TMP_README2="$(mktemp)"
cleanup() { rm -f "$INTEGRATED_FILE" "$TMP_README" "$TMP_README2"; }
trap cleanup EXIT

printf '%s' "$INTEGRATED" > "$INTEGRATED_FILE"

export RELEASE TARGET_PYTORCH PINNED_KINETO_COMMIT LAST_SYNC NEW_LAST_SYNC \
       LIBAIUPTI_VERSION AIU_TOOLKIT_VERSION PYTORCH_VERSION KINETO_SPYRE_VERSION

python3 - "$RECORD" "$INTEGRATED_FILE" <<'PY'
import json
import os
import sys

record_path, integrated_file = sys.argv[1], sys.argv[2]

with open(integrated_file, encoding="utf-8") as handle:
    integrated = [line.strip() for line in handle if line.strip()]

record = {
    "release": os.environ["RELEASE"],
    "target_pytorch": os.environ["TARGET_PYTORCH"],
    "pinned_kineto_commit": os.environ["PINNED_KINETO_COMMIT"],
    "previous_last_sync_commit": os.environ["LAST_SYNC"],
    "new_last_sync_commit": os.environ["NEW_LAST_SYNC"],
    "integrated_commits": integrated,
    "subcomponents": {
        "libaiupti": os.environ["LIBAIUPTI_VERSION"],
        "aiu_toolkit": os.environ["AIU_TOOLKIT_VERSION"],
        "pytorch": os.environ["PYTORCH_VERSION"],
        "kineto_spyre": os.environ["KINETO_SPYRE_VERSION"],
    },
}

with open(record_path, "w", encoding="utf-8") as handle:
    json.dump(record, handle, indent=2)
    handle.write("\n")
PY

echo "Wrote release record: $RECORD"
echo "  integrated commits: $(printf '%s\n' "$INTEGRATED" | grep -c . || true)"
echo "  new_last_sync_commit: $NEW_LAST_SYNC"

# ---- 4. Rewrite the README sync line (Req 1.5) -----------------------------
# Replace the 40-hex commit inside the backticked "last upstream sync" line.
sed "s/The last upstream sync was with the commit \`[0-9a-f]\{40\}\`/The last upstream sync was with the commit \`${NEW_LAST_SYNC}\`/" \
  "$README" > "$TMP_README"

# Reflect the PyTorch 2.12 target on a dedicated, idempotent line placed right
# after the sync line. Remove any prior target line first so re-runs do not
# accumulate duplicates.
TARGET_MM="$(printf '%s' "$TARGET_PYTORCH" | cut -d. -f1,2)"
TARGET_LINE="This release targets PyTorch \`${TARGET_MM}\`."

grep -v '^This release targets PyTorch ' "$TMP_README" > "$TMP_README2" || true

awk -v line="$TARGET_LINE" '
  { print }
  /^The last upstream sync was with the commit/ && !done { print line; done = 1 }
' "$TMP_README2" > "$README"

echo "Updated README sync line and PyTorch target in: $README"

# ---- 5. Validate the produced record --------------------------------------
python3 "$VALIDATOR" "$RECORD"

echo "record_sync.sh: done."
