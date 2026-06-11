#!/usr/bin/env bash
#
# resolve_pin.sh — Pin Resolver (release pipeline Stage A / design Component 1)
#
# Determine Pinned_Kineto_Commit for the kineto-spyre 2.12 release and verify it
# is a real, fetchable commit in upstream kineto.
#
#   * Read the `third_party/kineto` gitlink recorded at the PyTorch v2.12.0 tag
#     (via `git ls-tree`) to determine PINNED_KINETO_COMMIT — the commit SHA.
#     Upstream kineto is UNVERSIONED: PyTorch pins it to a plain main-branch
#     commit, so the pin is the 40-char SHA itself, NOT a release tag. We never
#     run `git describe --tags`.
#   * Verify that commit exists locally, or can be fetched, as a real commit
#     object in upstream kineto.
#
# Guards:
#   Req 2.5 — stop (non-zero) when the v2.12.0 tag has no third_party/kineto ref.
#   Req 2.6 — stop (non-zero) when the pinned commit cannot be found/fetched as a
#             real commit in upstream kineto.
#   Req 2.8 — the missing-reference stop is LOCAL: this script `exit`s its own
#             process with a non-zero status. It never signals or terminates any
#             sibling/parent process; other components may keep running.
#
# Output:
#   * `PINNED_KINETO_COMMIT=<sha>` on stdout (the only machine-readable line).
#   * If PIN_OUTPUT_FILE is set, the same `PINNED_KINETO_COMMIT=<sha>` line is
#     also written to that file.
#   * NO tag is emitted — the pin is the commit SHA only.
#   * All diagnostics go to stderr, so stdout stays clean for callers.
#
# Configuration (parameters / env vars, so the script is testable on fixtures):
#   PYTORCH_SRC          path to the PyTorch checkout (has the v2.12.0 tag).      [required]
#   UPSTREAM_KINETO      path to an upstream-kineto checkout to verify against.   [required]
#   PYTORCH_TAG          PyTorch tag to read the gitlink from. Default: v2.12.0.
#   KINETO_SUBMODULE     submodule path inside PyTorch. Default: third_party/kineto.
#   UPSTREAM_REMOTE_URL  URL/path used for the `upstream` remote when fetching.
#                        Default: https://github.com/pytorch/kineto.git
#   UPSTREAM_REMOTE_NAME name of the remote to (re)create. Default: upstream.
#   PIN_OUTPUT_FILE      optional file to also write PINNED_KINETO_COMMIT= to.
#
# Usage:
#   PYTORCH_SRC=/path/to/pytorch UPSTREAM_KINETO=/path/to/kineto ./scripts/resolve_pin.sh
#
set -euo pipefail

# --- configuration -----------------------------------------------------------
PYTORCH_SRC="${PYTORCH_SRC:-}"
UPSTREAM_KINETO="${UPSTREAM_KINETO:-}"
PYTORCH_TAG="${PYTORCH_TAG:-v2.12.0}"
KINETO_SUBMODULE="${KINETO_SUBMODULE:-third_party/kineto}"
UPSTREAM_REMOTE_URL="${UPSTREAM_REMOTE_URL:-https://github.com/pytorch/kineto.git}"
UPSTREAM_REMOTE_NAME="${UPSTREAM_REMOTE_NAME:-upstream}"
PIN_OUTPUT_FILE="${PIN_OUTPUT_FILE:-}"

err() { echo "ERROR: $*" >&2; }
info() { echo "$*" >&2; }

# --- argument / environment validation ---------------------------------------
if [ -z "$PYTORCH_SRC" ]; then
  err "PYTORCH_SRC is not set (path to the PyTorch checkout with the ${PYTORCH_TAG} tag)"
  exit 1
fi
if [ -z "$UPSTREAM_KINETO" ]; then
  err "UPSTREAM_KINETO is not set (path to an upstream-kineto checkout)"
  exit 1
fi
if [ ! -d "$PYTORCH_SRC" ]; then
  err "PYTORCH_SRC '$PYTORCH_SRC' is not a directory"
  exit 1
fi
if [ ! -d "$UPSTREAM_KINETO" ]; then
  err "UPSTREAM_KINETO '$UPSTREAM_KINETO' is not a directory"
  exit 1
fi

# --- 1. read the third_party/kineto gitlink at the PyTorch tag ----------------
# The gitlink is stored in the tag's tree object, so `git ls-tree` against the
# tag works even when the submodule is not checked out locally. We do NOT use
# `git describe --tags` — upstream kineto is unversioned; the pin is the SHA.

# Confirm the tag exists, with a clear message otherwise.
if ! git -C "$PYTORCH_SRC" rev-parse -q --verify "refs/tags/${PYTORCH_TAG}" >/dev/null 2>&1; then
  err "PyTorch checkout '$PYTORCH_SRC' has no tag '${PYTORCH_TAG}'"
  exit 1
fi

# Read the tree entry for the submodule path. A gitlink prints as:
#   160000 commit <sha>\t<path>
# An absent path prints nothing (and exits 0).
ls_tree_line="$(git -C "$PYTORCH_SRC" ls-tree "$PYTORCH_TAG" "$KINETO_SUBMODULE" || true)"

# Guard (Req 2.5): the tag must actually reference a commit at the submodule path.
if [ -z "$ls_tree_line" ]; then
  err "PyTorch ${PYTORCH_TAG} has no '${KINETO_SUBMODULE}' reference"
  # Req 2.8: stop ONLY this process (local non-zero exit); do not signal others.
  exit 1
fi

entry_type="$(printf '%s\n' "$ls_tree_line" | awk '{print $2}')"
PINNED_KINETO_COMMIT="$(printf '%s\n' "$ls_tree_line" | awk '{print $3}')"

# The reference must be a gitlink (a `commit` entry), not a tree/blob, and must
# be a real 40-char SHA. Otherwise the tag does not reference a kineto commit.
if [ "$entry_type" != "commit" ] || ! printf '%s' "$PINNED_KINETO_COMMIT" | grep -Eq '^[0-9a-f]{40}$'; then
  err "PyTorch ${PYTORCH_TAG} '${KINETO_SUBMODULE}' is not a commit reference (got: ${ls_tree_line})"
  exit 1   # Req 2.5: no usable commit reference -> stop before integration
fi

info "Pinned_Kineto_Commit = ${PINNED_KINETO_COMMIT}"

# --- 2. verify the commit is real / fetchable in upstream kineto --------------
# Upstream kineto is unversioned, so there is no tag to resolve — we only
# confirm the SHA exists and can be obtained as a commit object (Req 2.6).

commit_exists() {
  git -C "$UPSTREAM_KINETO" cat-file -e "${PINNED_KINETO_COMMIT}^{commit}" 2>/dev/null
}

if commit_exists; then
  info "Pinned_Kineto_Commit already present in upstream kineto"
else
  # Not present locally — (re)point the upstream remote and try to fetch exactly
  # that commit. Recreating the remote ensures we use the configured URL/path.
  info "Pinned_Kineto_Commit not local; attempting to fetch from '${UPSTREAM_REMOTE_NAME}' (${UPSTREAM_REMOTE_URL})"
  git -C "$UPSTREAM_KINETO" remote remove "$UPSTREAM_REMOTE_NAME" >/dev/null 2>&1 || true
  git -C "$UPSTREAM_KINETO" remote add "$UPSTREAM_REMOTE_NAME" "$UPSTREAM_REMOTE_URL"

  fetched=0
  if git -C "$UPSTREAM_KINETO" fetch --quiet "$UPSTREAM_REMOTE_NAME" "$PINNED_KINETO_COMMIT" 2>/dev/null; then
    fetched=1
  fi

  if [ "$fetched" -ne 1 ] || ! commit_exists; then
    err "${PINNED_KINETO_COMMIT} cannot be found or fetched as a commit in upstream kineto"
    exit 1   # Req 2.6: stop before integration
  fi
fi

info "Pinned_Kineto_Commit resolved and fetchable in upstream kineto"

# --- 3. emit the pin ----------------------------------------------------------
# stdout: the single machine-readable line. No tag output (the pin is the SHA).
echo "PINNED_KINETO_COMMIT=${PINNED_KINETO_COMMIT}"

if [ -n "$PIN_OUTPUT_FILE" ]; then
  echo "PINNED_KINETO_COMMIT=${PINNED_KINETO_COMMIT}" > "$PIN_OUTPUT_FILE"
  info "Wrote pin to ${PIN_OUTPUT_FILE}"
fi
