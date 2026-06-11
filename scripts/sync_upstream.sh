#!/usr/bin/env bash
#
# sync_upstream.sh — Component 2: Upstream Sync
#
# Cherry-pick the upstream kineto commit range from *after* Last_Sync_Commit
# (exclusive) through Pinned_Kineto_Commit (inclusive), applied in ascending
# chronological order, preserving empty (no-op) commits as empty commits and
# never integrating any commit newer than the pinned commit.
#
# On a cherry-pick conflict the script halts before any further pick, reports
# the conflicting upstream commit ID and the conflicted paths on stderr,
# retains every commit already integrated, and exits non-zero.
#
# Implements Requirements 1.1, 1.4, 1.7, 1.8, 2.4, 6.1, 6.2, 6.5.
#
# Parameters (positional args take precedence over environment):
#   $1 / LAST_SYNC               last synced upstream commit (range start, EXCLUSIVE)
#   $2 / PINNED_KINETO_COMMIT    pinned upstream commit (range end, INCLUSIVE)
#
# Environment overrides (all optional, parameterized for testability):
#   REPO_PATH        path to the kineto-spyre fork working tree (default: $PWD)
#   UPSTREAM_URL     upstream kineto remote URL
#                    (default: https://github.com/pytorch/kineto.git)
#   UPSTREAM_REMOTE  name of the upstream remote (default: upstream)
#
# Exit codes:
#   0  all commits in the range integrated successfully
#   2  cherry-pick halted on a merge conflict (tree left conflicted)
#   1  usage / pre-flight error (bad args, unknown commit, etc.)

set -euo pipefail

# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------
REPO_PATH="${REPO_PATH:-$PWD}"
LAST_SYNC="${1:-${LAST_SYNC:-}}"
PINNED_KINETO_COMMIT="${2:-${PINNED_KINETO_COMMIT:-}}"
UPSTREAM_URL="${UPSTREAM_URL:-https://github.com/pytorch/kineto.git}"
UPSTREAM_REMOTE="${UPSTREAM_REMOTE:-upstream}"

if [ -z "$LAST_SYNC" ] || [ -z "$PINNED_KINETO_COMMIT" ]; then
  echo "usage: sync_upstream.sh <LAST_SYNC> <PINNED_KINETO_COMMIT>" >&2
  echo "       (or set LAST_SYNC / PINNED_KINETO_COMMIT in the environment)" >&2
  exit 1
fi

# Run every git invocation against the target repository.
git_repo() { git -C "$REPO_PATH" "$@"; }

if ! git_repo rev-parse --git-dir >/dev/null 2>&1; then
  echo "ERROR: $REPO_PATH is not a git repository" >&2
  exit 1
fi

# --------------------------------------------------------------------------
# 1. Add the upstream kineto remote (idempotent) and fetch tags + history.
#    The URL is overridable so tests can point at a local fixture repo.
# --------------------------------------------------------------------------
if git_repo remote get-url "$UPSTREAM_REMOTE" >/dev/null 2>&1; then
  git_repo remote set-url "$UPSTREAM_REMOTE" "$UPSTREAM_URL"
else
  git_repo remote add "$UPSTREAM_REMOTE" "$UPSTREAM_URL"
fi
git_repo fetch "$UPSTREAM_REMOTE" --tags

# --------------------------------------------------------------------------
# Pre-flight: both range endpoints must resolve to real commits.
# --------------------------------------------------------------------------
for ref in "$LAST_SYNC" "$PINNED_KINETO_COMMIT"; do
  if ! git_repo cat-file -e "${ref}^{commit}" 2>/dev/null; then
    echo "ERROR: $ref cannot be resolved to a commit in $REPO_PATH" >&2
    exit 1
  fi
done

# --------------------------------------------------------------------------
# 2. Compute the cherry-pick range. "A..B" is EXCLUSIVE of A and INCLUSIVE of
#    B — exactly the "first exclusive, last inclusive" rule (Req 1.1). Because
#    the range stops at PINNED_KINETO_COMMIT, no commit newer than the pinned
#    commit is ever considered (Req 2.4). --reverse lists ascending so we can
#    report the integration order; cherry-pick of the range applies oldest
#    first regardless.
# --------------------------------------------------------------------------
RANGE="${LAST_SYNC}..${PINNED_KINETO_COMMIT}"
COMMIT_COUNT="$(git_repo rev-list --count "$RANGE")"
echo "Will integrate ${COMMIT_COUNT} commit(s) over range ${RANGE} (ascending)"

if [ "$COMMIT_COUNT" -eq 0 ]; then
  echo "Nothing to integrate: range ${RANGE} is empty."
  exit 0
fi

# --------------------------------------------------------------------------
# 3. Cherry-pick the whole range.
#    -x            records the source SHA in each message for provenance.
#    --empty=keep  preserves a no-op commit as an EMPTY commit instead of
#                  dropping it (Req 1.4). Change-producing commits take the
#                  default path and remain regular non-empty commits (Req 1.7).
#    A range cherry-pick applies oldest -> newest (ascending, Req 1.1) and
#    will NOT advance past a conflicted commit (Req 6.2).
# --------------------------------------------------------------------------
if git_repo cherry-pick -x --empty=keep "$RANGE"; then
  echo "Integrated ${COMMIT_COUNT} commit(s) up to and including ${PINNED_KINETO_COMMIT}"
  exit 0
fi

# --------------------------------------------------------------------------
# Conflict handling (Req 6.1, 6.2, 1.8).
#
# Conflict reporting takes PRIORITY over retention bookkeeping: capture and
# emit the conflicting upstream commit ID FIRST and independently, so that a
# retention failure can never suppress the conflict report (Req 1.8).
# --------------------------------------------------------------------------
CONFLICT_SHA=""
if CONFLICT_SHA="$(git_repo rev-parse --verify --quiet CHERRY_PICK_HEAD)"; then
  echo "CONFLICT while integrating upstream commit ${CONFLICT_SHA}" >&2
else
  # Not in a cherry-pick conflict state: the failure was something else.
  echo "ERROR: cherry-pick of range ${RANGE} failed without a conflict to report" >&2
  exit 1
fi

# Now (and only now) the secondary retention/path bookkeeping. Even if this
# section were to misbehave, the conflicting commit ID above is already on
# stderr (Req 1.8).
echo "Conflicted paths:" >&2
git_repo diff --name-only --diff-filter=U >&2 || true

# Req 6.1/6.2: every commit already integrated is retained (they are committed
# to the branch) and no further commit is picked while the tree is conflicted.
echo "Halting: resolve the conflict and run 'git cherry-pick --continue' to" >&2
echo "         resume integration in the original upstream commit order." >&2
exit 2
