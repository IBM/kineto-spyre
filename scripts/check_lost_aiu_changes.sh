#!/usr/bin/env bash
#
# check_lost_aiu_changes.sh — lost-AIU-change guard (design Component 3 /
# Error Handling).
#
# Run after a cherry-pick conflict has been resolved. It compares the paths
# the resolved cherry-pick commit (HEAD) actually changed against the paths the
# integrated upstream commit touches. If the resolution removed or overwrote a
# Kineto_Spyre AIU-specific path (e.g. under libkineto/src/plugin/aiupti/ or any
# other AIU/libaiupti code) that the integrated upstream commit does NOT modify,
# that AIU change was lost during conflict resolution: the guard reports the
# affected paths and stops (non-zero exit) so the sync does not continue.
#
# Requirements: 6.3 (retain AIU-specific changes not modified by the upstream
# commit), 6.4 (report a lost-change error naming the affected paths and do not
# continue).
#
# Usage:
#   check_lost_aiu_changes.sh <integrated-upstream-commit-sha> [repo-path]
#
# Environment:
#   REPO       repo path (overridden by the positional repo argument)
#   HEAD_REF   the resolved cherry-pick commit to inspect (default: HEAD)
#   AIU_PATHS  whitespace-separated substrings identifying AIU-specific paths
#              (default: "libkineto/src/plugin/aiupti aiupti libaiupti FindAIUToolkit")

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"

die() { echo "ERROR: $*" >&2; exit 1; }

UPSTREAM_SHA="${1:-}"
REPO="${2:-${REPO:-$DEFAULT_REPO}}"
HEAD_REF="${HEAD_REF:-HEAD}"
AIU_PATHS="${AIU_PATHS:-libkineto/src/plugin/aiupti aiupti libaiupti FindAIUToolkit}"

[ -n "$UPSTREAM_SHA" ] || die "usage: check_lost_aiu_changes.sh <integrated-upstream-commit-sha> [repo-path]"
[ -d "$REPO" ] || die "repo path does not exist: $REPO"

git -C "$REPO" cat-file -e "${UPSTREAM_SHA}^{commit}" 2>/dev/null \
  || die "integrated upstream commit not found in repo: $UPSTREAM_SHA"

# Paths the resolved cherry-pick (HEAD) changed, with status (M/D/A/R…).
HEAD_CHANGED="$(git -C "$REPO" show --name-status --pretty=format: "$HEAD_REF" | sed '/^$/d')"

# Paths the integrated upstream commit touches.
UPSTREAM_CHANGED="$(git -C "$REPO" show --name-only --pretty=format: "$UPSTREAM_SHA" | sed '/^$/d')"

is_aiu_path() {
  local path="$1" pat
  for pat in $AIU_PATHS; do
    case "$path" in
      *"$pat"*) return 0;;
    esac
  done
  return 1
}

lost=()
while IFS="$(printf '\t')" read -r status path rest; do
  [ -n "$path" ] || continue
  # Only AIU-specific paths are guarded (Req 6.3).
  is_aiu_path "$path" || continue
  # "Removed or overwritten" => deleted (D), modified (M), or renamed (R).
  case "$status" in
    D*|M*|R*) : ;;
    *) continue;;
  esac
  # If the integrated upstream commit also touches this path, the change is a
  # legitimate part of the integration — not a lost AIU change.
  if printf '%s\n' "$UPSTREAM_CHANGED" | grep -Fxq "$path"; then
    continue
  fi
  lost+=("$path")
done <<EOF
$HEAD_CHANGED
EOF

if [ "${#lost[@]}" -gt 0 ]; then
  echo "ERROR: lost AIU-specific change(s) during conflict resolution of upstream commit ${UPSTREAM_SHA}." >&2
  echo "       The following AIU path(s) were removed/overwritten but are NOT modified by that upstream commit:" >&2
  for p in "${lost[@]}"; do
    echo "         - $p" >&2
  done
  echo "       Restore these changes before continuing the sync (Req 6.4)." >&2
  exit 1
fi

echo "OK: no AIU-specific changes were lost resolving upstream commit ${UPSTREAM_SHA}."
