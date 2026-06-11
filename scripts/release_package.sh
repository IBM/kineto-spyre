#!/usr/bin/env bash
#
# release_package.sh — kineto-spyre 2.12 release tagging + packaging (Component 7)
#
# Implements Requirement 10 (Release Tagging and Packaging):
#   10.1  After the wheel is built AND the smoke test passed, create exactly ONE
#         release tag whose name identifies the release as PyTorch 2.12.
#   10.2  The artifact set includes exactly ONE built PyTorch wheel.
#   10.3  The artifact set includes the README updated with the synced upstream
#         kineto commit ID and the PyTorch 2.12 target.
#   10.4  Every recorded integrated upstream commit ID is associated with the tag
#         (via release_record.json committed at the tagged revision + release notes).
#   10.5  If a 2.12 tag already exists, report a tagging conflict and DO NOT
#         overwrite it.
#   10.6  If tag creation fails, report the failure and publish nothing.
#
# ---------------------------------------------------------------------------
# TAG NAMING vs WHEEL VERSION — IMPORTANT DISTINCTION
# ---------------------------------------------------------------------------
# This repo's git tags / GitHub release names use the historical DOTTED form
# that matches every prior release tag in this repo, e.g.:
#       torch-2.11.0.aiu.kineto.1.1.2
#       torch-2.10.0.aiu.kineto.1.1.2
# so the 2.12 release tag is:
#       torch-2.12.0.aiu.kineto.<x.y.z>      e.g. torch-2.12.0.aiu.kineto.1.2.0
# (DOTS between "torch-<pytorch>" and "aiu.kineto.<ver>", NOT a '+').
#
# The PEP 440 *wheel* version produced by build_pytorch.sh is a SEPARATE
# artifact and still uses the local-version '+' form required by PEP 440:
#       torch-2.12.0+aiu.kineto.<x.y.z>-cp312-cp312-linux_<arch>.whl
# i.e. the '+' belongs to the wheel filename / PYTORCH_BUILD_VERSION only; the
# git TAG and the GitHub release NAME use the dotted form. Do not conflate them.
# ---------------------------------------------------------------------------
#
# This script is built to be TESTABLE OFFLINE. Everything that could touch a
# real remote or the network is parameterized and/or guarded:
#   * REMOTE points at any git remote — tests point it at a local *bare* repo.
#   * Publishing the GitHub release goes through publish_release(), which is a
#     no-op under DRY_RUN=1 (the default) and otherwise calls ${GH_BIN:-gh}.
#     Tests never set DRY_RUN=0, so `gh`/network are never invoked.
#
# The contributor account has READ-ONLY access to IBM/kineto-spyre, so this
# script is authored + exercised against fixture git repos in temp dirs only.

set -euo pipefail

# ---------------------------------------------------------------------------
# Parameters (all overridable via environment for testing)
# ---------------------------------------------------------------------------
PYTORCH_VERSION="${PYTORCH_VERSION:-2.12.0}"   # Target_PyTorch release (Req 2.3, 10.3)
KINETO_VERSION="${KINETO_VERSION:-}"            # kineto-spyre subcomponent version, e.g. 1.2.0

# Repo to operate in (the kineto-spyre fork checkout). Defaults to CWD.
REPO_PATH="${REPO_PATH:-$(pwd)}"

# Remote to push the tag to. In tests this is a path to a local bare repo so
# nothing ever hits a real network remote.
REMOTE="${REMOTE:-origin}"

# Where the built wheel lives (PyTorch's dist/ by default).
WHEEL_DIR="${WHEEL_DIR:-${REPO_PATH}/dist}"

# Provenance + docs (relative to REPO_PATH unless absolute).
RELEASE_RECORD="${RELEASE_RECORD:-${REPO_PATH}/release_record.json}"
README_PATH="${README_PATH:-${REPO_PATH}/README.md}"

# Staging dir where the offline-assembled Release_Artifact_Set is collected.
STAGING_DIR="${STAGING_DIR:-${REPO_PATH}/release_staging}"

# Tag name. Derived from the DOTTED convention unless explicitly overridden.
# Capture whether TAG was supplied via the environment before defaulting it.
if [ -n "${TAG:-}" ]; then TAG_OVERRIDDEN=1; else TAG_OVERRIDDEN=0; fi
TAG="${TAG:-torch-${PYTORCH_VERSION}.aiu.kineto.${KINETO_VERSION}}"

# Wheel filename version (the PEP 440 '+' local-version form) — informational,
# used only for documentation/notes, never for the tag.
WHEEL_VERSION="${PYTORCH_VERSION}+aiu.kineto.${KINETO_VERSION}"

# Preconditions: smoke result + push toggle + publish toggle.
#   SMOKE_PASSED must be explicitly "1" (fail-closed) — the tag is created only
#   AFTER the wheel was built and the smoke test passed (Req 10.1).
SMOKE_PASSED="${SMOKE_PASSED:-0}"
PUSH_TAG="${PUSH_TAG:-1}"        # push the tag to REMOTE (Req 10.1)
DRY_RUN="${DRY_RUN:-1}"          # 1 => never call real `gh`/network (default, test-safe)
GH_BIN="${GH_BIN:-gh}"           # GitHub CLI binary (stubbable in tests)

log()  { echo "[release_package] $*"; }
err()  { echo "[release_package] ERROR: $*" >&2; }

git_repo() { git -C "$REPO_PATH" "$@"; }

# ---------------------------------------------------------------------------
# Preconditions (fail closed) — Req 10.1
# ---------------------------------------------------------------------------

require_inputs() {
  if [ -z "$KINETO_VERSION" ] && [ "${TAG_OVERRIDDEN:-0}" != "1" ]; then
    err "KINETO_VERSION is required (used to build the tag name)"
    return 1
  fi
  if [ ! -d "${REPO_PATH}/.git" ]; then
    err "REPO_PATH '$REPO_PATH' is not a git repository"
    return 1
  fi
}

# Req 10.1: the tag is created only AFTER the smoke test passed. We treat
# "smoke passed" as a precondition the caller asserts via SMOKE_PASSED=1.
require_smoke_passed() {
  if [ "$SMOKE_PASSED" != "1" ]; then
    err "smoke test has not passed (SMOKE_PASSED != 1) — refusing to tag/publish"
    return 1
  fi
  log "precondition: smoke test passed"
}

# Req 10.1 / 10.2: exactly one wheel must exist. Echoes the wheel path on stdout.
find_single_wheel() {
  local wheels=()
  if [ -d "$WHEEL_DIR" ]; then
    # Collect *.whl without relying on nullglob being set globally.
    while IFS= read -r -d '' w; do
      wheels+=("$w")
    done < <(find "$WHEEL_DIR" -maxdepth 1 -name '*.whl' -print0 2>/dev/null)
  fi
  local n=${#wheels[@]}
  if [ "$n" -ne 1 ]; then
    err "expected exactly 1 wheel in '$WHEEL_DIR', found $n"
    return 1
  fi
  printf '%s\n' "${wheels[0]}"
}

# ---------------------------------------------------------------------------
# Provenance helpers — Req 10.3, 10.4
# ---------------------------------------------------------------------------

# Read a top-level string field from release_record.json without requiring jq.
record_field() {
  local field="$1"
  python3 - "$RELEASE_RECORD" "$field" <<'PY'
import json, sys
path, field = sys.argv[1], sys.argv[2]
with open(path) as fh:
    data = json.load(fh)
val = data.get(field, "")
if val is None:
    val = ""
print(val)
PY
}

# Echo every integrated upstream commit ID, one per line (ascending).
record_integrated_commits() {
  python3 - "$RELEASE_RECORD" <<'PY'
import json, sys
with open(sys.argv[1]) as fh:
    data = json.load(fh)
for sha in data.get("integrated_commits", []):
    print(sha)
PY
}

# Req 10.3: verify the README has been updated with the synced upstream commit
# ID and the PyTorch 2.12 target. record_sync.sh (PR 1) performs the rewrite;
# here we VERIFY/ASSERT it, failing closed if the README was not updated.
verify_readme_updated() {
  local synced_commit="$1"
  if [ ! -f "$README_PATH" ]; then
    err "README not found at '$README_PATH'"
    return 1
  fi
  if ! grep -q "$synced_commit" "$README_PATH"; then
    err "README does not reference the synced upstream commit $synced_commit (Req 10.3)"
    return 1
  fi
  # Match the PyTorch 2.12 target (e.g. "2.12" or "2.12.0").
  if ! grep -Eq "2\.12(\.[0-9]+)?" "$README_PATH"; then
    err "README does not reference the PyTorch 2.12 target (Req 10.3)"
    return 1
  fi
  log "README verified: references synced commit $synced_commit and PyTorch 2.12"
}

# ---------------------------------------------------------------------------
# Tagging — Req 10.1, 10.4, 10.5, 10.6
# ---------------------------------------------------------------------------

# Req 10.5: refuse to overwrite an existing tag.
assert_tag_absent() {
  if git_repo rev-parse -q --verify "refs/tags/${TAG}" >/dev/null 2>&1; then
    err "tagging conflict: tag '${TAG}' already exists — refusing to overwrite (Req 10.5)"
    return 1
  fi
}

# Req 10.4: ensure release_record.json (and the updated README) are committed so
# the tag points at a revision that carries the integrated-commit provenance.
commit_provenance_if_needed() {
  # Stage record + README if they are tracked-with-changes or untracked.
  local changed=0
  git_repo add -- "$RELEASE_RECORD" "$README_PATH" 2>/dev/null || true
  if ! git_repo diff --cached --quiet 2>/dev/null; then
    changed=1
  fi
  if [ "$changed" -eq 1 ]; then
    git_repo commit -m "Record provenance for ${TAG}" >/dev/null
    log "committed provenance (release_record.json + README) for ${TAG}"
  else
    log "provenance already committed; tagging current HEAD"
  fi
}

# Req 10.1 / 10.6: create exactly one annotated tag; on failure, stop and
# publish nothing. The annotation embeds the pinned commit for auditability.
create_tag() {
  local pinned_commit="$1"
  if ! git_repo tag -a "${TAG}" \
        -m "kineto-spyre ${PYTORCH_VERSION} release (wheel ${WHEEL_VERSION}); upstream kineto pinned at ${pinned_commit}"; then
    err "tag creation failed for '${TAG}' — publishing nothing (Req 10.6)"
    return 1
  fi
  log "created annotated tag ${TAG}"
}

# Req 10.1: push the tag to the (parameterized) remote.
push_tag() {
  if [ "$PUSH_TAG" != "1" ]; then
    log "PUSH_TAG != 1 — skipping push"
    return 0
  fi
  if ! git_repo push "$REMOTE" "refs/tags/${TAG}"; then
    err "failed to push tag '${TAG}' to remote '${REMOTE}'"
    return 1
  fi
  log "pushed tag ${TAG} to ${REMOTE}"
}

# ---------------------------------------------------------------------------
# Artifact-set assembly + publish — Req 10.2, 10.3, 10.4
# ---------------------------------------------------------------------------

# Build the offline Release_Artifact_Set in STAGING_DIR: the single wheel, the
# updated README, release_record.json, and release notes that embed/link the
# record so every integrated commit ID is associated with the tag.
assemble_artifact_set() {
  local wheel="$1" pinned_commit="$2" new_last_sync="$3"

  rm -rf "$STAGING_DIR"
  mkdir -p "$STAGING_DIR"

  cp "$wheel" "$STAGING_DIR/"
  cp "$README_PATH" "$STAGING_DIR/README.md"
  cp "$RELEASE_RECORD" "$STAGING_DIR/release_record.json"

  local notes="${STAGING_DIR}/RELEASE_NOTES.md"
  {
    echo "# kineto-spyre ${PYTORCH_VERSION} release"
    echo
    echo "- **Release tag:** \`${TAG}\`"
    echo "- **PyTorch wheel version:** \`${WHEEL_VERSION}\` (\`$(basename "$wheel")\`)"
    echo "- **PyTorch target:** ${PYTORCH_VERSION}"
    echo "- **Pinned upstream kineto commit:** \`${pinned_commit}\`"
    echo "- **New last upstream sync commit:** \`${new_last_sync}\`"
    echo
    echo "Full provenance is recorded in the committed \`release_record.json\`"
    echo "(included in this release and present at the tagged revision)."
    echo
    echo "## Integrated upstream kineto commits"
    echo
    local any=0
    while IFS= read -r sha; do
      [ -n "$sha" ] || continue
      echo "- \`${sha}\`"
      any=1
    done < <(record_integrated_commits)
    if [ "$any" -eq 0 ]; then
      echo "_(none recorded)_"
    fi
  } > "$notes"

  log "assembled artifact set in ${STAGING_DIR}:"
  log "  wheel:           $(basename "$wheel")"
  log "  README.md"
  log "  release_record.json"
  log "  RELEASE_NOTES.md"
}

# Create the GitHub release. Guarded so tests never touch the network:
#   * DRY_RUN=1 (default): print the command that WOULD run, then return.
#   * DRY_RUN=0: invoke ${GH_BIN} (stubbable). Only used outside tests.
publish_release() {
  local wheel="$1"
  local notes="${STAGING_DIR}/RELEASE_NOTES.md"

  local -a cmd=(
    "$GH_BIN" release create "$TAG"
    --title "$TAG"
    --notes-file "$notes"
    "$wheel"
    "${STAGING_DIR}/release_record.json"
  )

  if [ "$DRY_RUN" = "1" ]; then
    log "DRY_RUN=1 — not publishing. Would run:"
    log "  ${cmd[*]}"
    return 0
  fi

  if ! command -v "$GH_BIN" >/dev/null 2>&1; then
    err "GitHub CLI '${GH_BIN}' not available — cannot publish release"
    return 1
  fi
  log "publishing GitHub release ${TAG}"
  ( cd "$REPO_PATH" && "${cmd[@]}" )
}

# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------
main() {
  require_inputs
  require_smoke_passed                              # Req 10.1 precondition

  local wheel
  wheel="$(find_single_wheel)"                      # Req 10.1/10.2 precondition
  log "found wheel: $wheel"

  if [ ! -f "$RELEASE_RECORD" ]; then
    err "release_record.json not found at '$RELEASE_RECORD' (Req 10.4)"
    return 1
  fi

  local pinned_commit new_last_sync
  pinned_commit="$(record_field pinned_kineto_commit)"
  new_last_sync="$(record_field new_last_sync_commit)"
  [ -n "$new_last_sync" ] || new_last_sync="$pinned_commit"

  verify_readme_updated "$new_last_sync"            # Req 10.3

  # Req 10.5 BEFORE any tagging/publishing: never overwrite an existing tag, and
  # never stage/publish artifacts when the tag cannot be created cleanly.
  assert_tag_absent

  commit_provenance_if_needed                       # Req 10.4 (record at tagged rev)
  create_tag "$pinned_commit"                       # Req 10.1, 10.6
  push_tag                                          # Req 10.1

  # Publishing happens ONLY after the tag was created successfully (Req 10.6).
  assemble_artifact_set "$wheel" "$pinned_commit" "$new_last_sync"   # Req 10.2/10.3/10.4
  publish_release "$wheel"

  log "release ${TAG} prepared successfully"
}

# Allow the file to be sourced (for unit-testing individual functions) without
# running main. When executed directly, run main.
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  main "$@"
fi
