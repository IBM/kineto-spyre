#!/usr/bin/env python3
"""Tests for ``scripts/resolve_pin.sh`` (Pin Resolver, design Component 1).

These drive the real bash script via ``subprocess`` against small fixture git
repositories built in temp dirs. They cover:

* Task 2.2 (integration, Req 2.1, 2.2): a fixture PyTorch repo whose
  ``third_party/kineto`` gitlink points at a known SHA, plus a fixture upstream
  kineto repo that actually contains that commit -> the script resolves the
  expected SHA and verifies it is fetchable.
* Task 2.3 (guards, Req 2.5, 2.6, 2.8): a fixture where the ``v2.12.0`` tag has
  no ``third_party/kineto`` reference -> stop, non-zero; a fixture whose gitlink
  points at a SHA absent from upstream kineto -> stop, non-zero (unfetchable);
  and assertions that the stop is a *local* non-zero process exit that does not
  terminate the test runner / parent (Req 2.8).

Run from the repo root with::

    python3 -m unittest scripts.test_resolve_pin

or from this directory with::

    python3 -m unittest test_resolve_pin
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest

SCRIPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resolve_pin.sh")

# A syntactically valid 40-char SHA that will not exist in any fixture repo.
ABSENT_SHA = "deadbeef" * 5  # 40 hex chars

# Deterministic identity + isolation for fixture git operations.
GIT_ENV = {
    "GIT_AUTHOR_NAME": "Test",
    "GIT_AUTHOR_EMAIL": "test@example.com",
    "GIT_COMMITTER_NAME": "Test",
    "GIT_COMMITTER_EMAIL": "test@example.com",
    # Keep fixtures hermetic: ignore any global/system git config.
    "GIT_CONFIG_NOSYSTEM": "1",
    "HOME": tempfile.gettempdir(),
}


def git(repo: str, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a git command inside ``repo``."""
    env = os.environ.copy()
    env.update(GIT_ENV)
    return subprocess.run(
        ["git", "-C", repo, *args],
        check=check,
        capture_output=True,
        text=True,
        env=env,
    )


def init_repo(path: str) -> None:
    os.makedirs(path, exist_ok=True)
    git(path, "init", "-q", "-b", "main")
    git(path, "config", "user.email", "test@example.com")
    git(path, "config", "user.name", "Test")


def commit_file(repo: str, name: str, content: str, message: str) -> str:
    """Create/overwrite a file, commit it, and return the new commit SHA."""
    with open(os.path.join(repo, name), "w", encoding="utf-8") as handle:
        handle.write(content)
    git(repo, "add", name)
    git(repo, "commit", "-q", "-m", message)
    return git(repo, "rev-parse", "HEAD").stdout.strip()


def add_gitlink(repo: str, submodule_path: str, sha: str) -> None:
    """Add a gitlink (submodule pointer) at ``submodule_path`` -> ``sha``."""
    git(repo, "update-index", "--add", "--cacheinfo", f"160000,{sha},{submodule_path}")


def run_resolve_pin(pytorch_src: str, upstream_kineto: str, **extra_env):
    """Invoke resolve_pin.sh and return the CompletedProcess."""
    env = os.environ.copy()
    env.update(GIT_ENV)
    env["PYTORCH_SRC"] = pytorch_src
    env["UPSTREAM_KINETO"] = upstream_kineto
    env.update({k: str(v) for k, v in extra_env.items()})
    return subprocess.run(
        ["bash", SCRIPT_PATH],
        capture_output=True,
        text=True,
        env=env,
    )


class ResolvePinTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="resolve_pin_test_")
        self.pytorch = os.path.join(self.tmp, "pytorch")
        self.upstream = os.path.join(self.tmp, "kineto")
        init_repo(self.pytorch)
        init_repo(self.upstream)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def seed_upstream_with_commits(self) -> str:
        """Seed the upstream fixture with a few commits; return the last SHA."""
        commit_file(self.upstream, "a.txt", "1", "first upstream commit")
        commit_file(self.upstream, "b.txt", "2", "second upstream commit")
        return commit_file(self.upstream, "c.txt", "3", "pinned upstream commit")

    def make_pytorch_tag_with_gitlink(self, sha: str, tag: str = "v2.12.0") -> None:
        """Create a PyTorch fixture whose tag pins third_party/kineto -> sha."""
        commit_file(self.pytorch, "version.txt", "2.12.0\n", "pytorch base")
        add_gitlink(self.pytorch, "third_party/kineto", sha)
        git(self.pytorch, "commit", "-q", "-m", "pin kineto submodule")
        git(self.pytorch, "tag", tag)


class IntegrationTest(ResolvePinTestBase):
    """Task 2.2 — Req 2.1, 2.2."""

    def test_resolves_expected_sha_and_verifies_fetchable(self):
        pinned = self.seed_upstream_with_commits()
        self.make_pytorch_tag_with_gitlink(pinned)

        result = run_resolve_pin(self.pytorch, self.upstream)

        self.assertEqual(
            result.returncode, 0,
            msg=f"expected success.\nstdout={result.stdout}\nstderr={result.stderr}",
        )
        # stdout carries exactly the machine-readable pin line, no tag output.
        self.assertIn(f"PINNED_KINETO_COMMIT={pinned}", result.stdout)
        self.assertNotIn("TAG", result.stdout.upper())
        # The commit is present locally in upstream, so it is verified fetchable.
        self.assertIn("resolved and fetchable", result.stderr)

    def test_writes_pin_to_output_file(self):
        pinned = self.seed_upstream_with_commits()
        self.make_pytorch_tag_with_gitlink(pinned)
        out_file = os.path.join(self.tmp, "pin.txt")

        result = run_resolve_pin(self.pytorch, self.upstream, PIN_OUTPUT_FILE=out_file)

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        with open(out_file, encoding="utf-8") as handle:
            self.assertEqual(handle.read().strip(), f"PINNED_KINETO_COMMIT={pinned}")


class GuardTest(ResolvePinTestBase):
    """Task 2.3 — Req 2.5, 2.6, 2.8."""

    def test_missing_kineto_reference_stops_nonzero(self):
        # Req 2.5: tag exists but has no third_party/kineto reference.
        commit_file(self.pytorch, "version.txt", "2.12.0\n", "pytorch base")
        git(self.pytorch, "tag", "v2.12.0")

        result = run_resolve_pin(self.pytorch, self.upstream)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no 'third_party/kineto' reference", result.stderr)
        # No pin emitted on stdout when stopping before integration.
        self.assertNotIn("PINNED_KINETO_COMMIT=", result.stdout)

    def test_missing_reference_stop_is_local_not_a_signal(self):
        # Req 2.8: the stop is a LOCAL non-zero process exit (via `exit`), not a
        # signal that would terminate the parent/test runner. A subprocess killed
        # by a signal reports a negative returncode in Python; a normal local
        # exit reports a positive status. The test runner itself keeps going
        # (this assertion executing at all proves the parent was not terminated).
        commit_file(self.pytorch, "version.txt", "2.12.0\n", "pytorch base")
        git(self.pytorch, "tag", "v2.12.0")

        result = run_resolve_pin(self.pytorch, self.upstream)

        self.assertGreater(
            result.returncode, 0,
            msg="missing-reference stop must be a local non-zero exit, not a signal",
        )

    def test_pinned_commit_absent_from_upstream_stops_nonzero(self):
        # Req 2.6: gitlink points at a SHA that upstream kineto does not contain
        # and cannot fetch. Point the upstream remote at a local repo (the
        # upstream fixture itself) so no network access is attempted; that repo
        # does not contain ABSENT_SHA, so the fetch fails cleanly.
        self.seed_upstream_with_commits()
        self.make_pytorch_tag_with_gitlink(ABSENT_SHA)

        result = run_resolve_pin(
            self.pytorch,
            self.upstream,
            UPSTREAM_REMOTE_URL=self.upstream,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertGreater(result.returncode, 0, msg="must be a local non-zero exit")
        self.assertIn("cannot be found or fetched", result.stderr)
        self.assertNotIn("PINNED_KINETO_COMMIT=", result.stdout)


if __name__ == "__main__":
    unittest.main()
