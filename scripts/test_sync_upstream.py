#!/usr/bin/env python3
"""Tests for ``scripts/sync_upstream.sh`` (Component 2: Upstream Sync).

These tests drive the real bash script via ``subprocess`` against throwaway
fixture git repositories built with ``git init`` and real commits. No network
access and no live repositories are touched — the upstream remote URL is
overridden to point at a local fixture repo.

Run from the repo root with::

    python3 -m unittest scripts.test_sync_upstream

or from this directory with::

    python3 -m unittest test_sync_upstream

Covers task 3.3 (integration: integrated set/order, conflict halt + reporting,
ordered continuation) and task 3.4 (edge cases: empty-commit preservation,
exclusive/inclusive range boundary, no commit newer than the pinned commit).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import unittest

SCRIPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sync_upstream.sh")

# Deterministic identity + environment so commits are reproducible and no
# interactive editor / signing ever blocks a cherry-pick.
GIT_ENV = {
    "GIT_AUTHOR_NAME": "Test",
    "GIT_AUTHOR_EMAIL": "test@example.com",
    "GIT_COMMITTER_NAME": "Test",
    "GIT_COMMITTER_EMAIL": "test@example.com",
    "GIT_AUTHOR_DATE": "2020-01-01T00:00:00",
    "GIT_COMMITTER_DATE": "2020-01-01T00:00:00",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_SYSTEM": "/dev/null",
    "EDITOR": "true",
    "GIT_EDITOR": "true",
}


def _full_env() -> dict:
    env = os.environ.copy()
    env.update(GIT_ENV)
    return env


def git(repo: str, *args: str) -> str:
    """Run a git command in ``repo`` and return stripped stdout."""
    result = subprocess.run(
        ["git", "-C", repo, *args],
        env=_full_env(),
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


class GitFixture:
    """A throwaway git repository with helpers to build a known history."""

    def __init__(self, path: str):
        self.path = path

    @classmethod
    def init(cls, path: str) -> "GitFixture":
        os.makedirs(path, exist_ok=True)
        subprocess.run(["git", "init", "-q", "-b", "main", path], env=_full_env(), check=True)
        git(path, "config", "commit.gpgsign", "false")
        git(path, "config", "user.name", "Test")
        git(path, "config", "user.email", "test@example.com")
        return cls(path)

    def write(self, relpath: str, content: str) -> None:
        full = os.path.join(self.path, relpath)
        os.makedirs(os.path.dirname(full) or self.path, exist_ok=True)
        with open(full, "w", encoding="utf-8") as handle:
            handle.write(content)

    def commit_file(self, relpath: str, content: str, message: str) -> str:
        self.write(relpath, content)
        git(self.path, "add", relpath)
        git(self.path, "commit", "-q", "-m", message)
        return self.head()

    def commit_empty(self, message: str) -> str:
        git(self.path, "commit", "-q", "--allow-empty", "-m", message)
        return self.head()

    def head(self) -> str:
        return git(self.path, "rev-parse", "HEAD")


def run_sync(fork: str, upstream: str, last_sync: str, pinned: str) -> subprocess.CompletedProcess:
    """Invoke sync_upstream.sh against the fork, pointing upstream at a fixture."""
    env = _full_env()
    env.update(
        {
            "REPO_PATH": fork,
            "UPSTREAM_URL": upstream,
            "UPSTREAM_REMOTE": "upstream",
        }
    )
    return subprocess.run(
        ["bash", SCRIPT_PATH, last_sync, pinned],
        env=env,
        capture_output=True,
        text=True,
    )


def integrated_provenance_shas(fork: str, base: str) -> list:
    """Return upstream SHAs recorded by ``-x`` in commits after ``base``, ascending."""
    body = git(fork, "log", "--reverse", "--pretty=%B", f"{base}..HEAD")
    return re.findall(r"cherry picked from commit ([0-9a-f]{40})", body)


def is_empty_commit(fork: str, commit: str) -> bool:
    """True when ``commit``'s tree equals its first parent's tree (a no-op)."""
    tree = git(fork, "rev-parse", f"{commit}^{{tree}}")
    parent_tree = git(fork, "rev-parse", f"{commit}^^{{tree}}")
    return tree == parent_tree


class SyncUpstreamTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="sync_upstream_test_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.upstream_path = os.path.join(self.tmp, "upstream")
        self.fork_path = os.path.join(self.tmp, "fork")


class HappyPathTest(SyncUpstreamTestBase):
    """Task 3.3/3.4: integrated set/order, empty preservation, range boundary."""

    def setUp(self) -> None:
        super().setUp()
        up = GitFixture.init(self.upstream_path)
        # C0 = Last_Sync_Commit (range start, EXCLUSIVE).
        self.c0 = up.commit_file("common.txt", "original\n", "C0 base (last sync)")
        # C1 = change-producing (adds a file) -> must be a regular non-empty commit.
        self.c1 = up.commit_file("feature1.txt", "f1\n", "C1 add feature1")
        # C2 = no-op / empty commit -> must be preserved as an empty commit.
        self.c2 = up.commit_empty("C2 empty no-op")
        # C3 = PINNED (range end, INCLUSIVE), change-producing.
        self.c3 = up.commit_file("feature2.txt", "f2\n", "C3 add feature2 (pinned)")
        # C4 = beyond the pinned commit -> must NOT be integrated.
        self.c4 = up.commit_file("feature3.txt", "f3\n", "C4 add feature3 (beyond pin)")

        # Independent fork (no shared history) with an AIU-specific file.
        fork = GitFixture.init(self.fork_path)
        self.fork_base = fork.commit_file(
            "aiu_plugin.txt", "aiu specific code\n", "fork base"
        )

    def test_integrates_expected_set_and_order(self):
        proc = run_sync(self.fork_path, self.upstream_path, self.c0, self.c3)
        self.assertEqual(proc.returncode, 0, proc.stderr)

        shas = integrated_provenance_shas(self.fork_path, self.fork_base)
        # Req 1.1: ascending order, C0 excluded, C3 included; C4 excluded (Req 2.4).
        self.assertEqual(shas, [self.c1, self.c2, self.c3])

    def test_range_boundary_excludes_last_sync_and_beyond_pinned(self):
        proc = run_sync(self.fork_path, self.upstream_path, self.c0, self.c3)
        self.assertEqual(proc.returncode, 0, proc.stderr)

        shas = integrated_provenance_shas(self.fork_path, self.fork_base)
        # Req 1.1 (exclusive start): C0 is never re-integrated.
        self.assertNotIn(self.c0, shas)
        # Req 2.4: nothing newer than the pinned commit is integrated.
        self.assertNotIn(self.c4, shas)
        # The beyond-pin file must be absent from the fork tree.
        self.assertFalse(os.path.exists(os.path.join(self.fork_path, "feature3.txt")))
        # The included files are present.
        self.assertTrue(os.path.exists(os.path.join(self.fork_path, "feature1.txt")))
        self.assertTrue(os.path.exists(os.path.join(self.fork_path, "feature2.txt")))

    def test_change_producing_commit_is_regular_non_empty(self):
        proc = run_sync(self.fork_path, self.upstream_path, self.c0, self.c3)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        # Req 1.7: the commit integrating C1 carries its file change (non-empty).
        c1_commit = self._fork_commit_for(self.c1)
        self.assertFalse(is_empty_commit(self.fork_path, c1_commit))

    def test_empty_commit_preserved_not_skipped(self):
        proc = run_sync(self.fork_path, self.upstream_path, self.c0, self.c3)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        # Req 1.4: the no-op commit C2 is kept as an EMPTY commit, never skipped.
        shas = integrated_provenance_shas(self.fork_path, self.fork_base)
        self.assertIn(self.c2, shas, "empty commit was dropped instead of preserved")
        c2_commit = self._fork_commit_for(self.c2)
        self.assertTrue(
            is_empty_commit(self.fork_path, c2_commit),
            "C2 should be preserved as an empty commit",
        )

    def _fork_commit_for(self, upstream_sha: str) -> str:
        """Find the fork commit whose -x line references ``upstream_sha``."""
        out = git(
            self.fork_path,
            "log",
            "--pretty=%H%x1f%B%x1e",
            f"{self.fork_base}..HEAD",
        )
        for record in out.split("\x1e"):
            record = record.strip()
            if not record:
                continue
            sha, _, body = record.partition("\x1f")
            if upstream_sha in body:
                return sha.strip()
        self.fail(f"no fork commit references upstream {upstream_sha}")


class ConflictTest(SyncUpstreamTestBase):
    """Task 3.3: conflict halt + reporting, retention, ordered continuation."""

    def setUp(self) -> None:
        super().setUp()
        up = GitFixture.init(self.upstream_path)
        # C0 = Last_Sync_Commit.
        self.c0 = up.commit_file("common.txt", "original\n", "C0 base (last sync)")
        # C1 = clean change (integrates before the conflict; must be retained).
        self.c1 = up.commit_file("feature1.txt", "f1\n", "C1 add feature1")
        # C2 = modifies common.txt -> conflicts with the fork-local change.
        self.c2 = up.commit_file("common.txt", "upstream-modified\n", "C2 modify common")
        # C3 = PINNED, clean change after the conflict (only after resolution).
        self.c3 = up.commit_file("feature2.txt", "f2\n", "C3 add feature2 (pinned)")

        fork = GitFixture.init(self.fork_path)
        fork.commit_file("aiu_plugin.txt", "aiu specific code\n", "fork base")
        # Fork-local divergent change to the same line C2 touches -> conflict.
        self.fork_base = fork.commit_file(
            "common.txt", "fork-modified\n", "fork local change to common"
        )

    def test_conflict_halts_reports_and_retains(self):
        proc = run_sync(self.fork_path, self.upstream_path, self.c0, self.c3)

        # Req 6.2: non-zero exit; the script halts on the conflict.
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        # Req 6.1 / 1.8: the conflicting upstream SHA is reported on stderr.
        self.assertIn(self.c2, proc.stderr)
        # Req 6.1: the conflicted path is reported on stderr.
        self.assertIn("common.txt", proc.stderr)
        # The conflicting SHA is reported BEFORE the conflicted-paths bookkeeping
        # (Req 1.8: conflict reporting takes priority over retention).
        self.assertLess(
            proc.stderr.index(self.c2),
            proc.stderr.index("Conflicted paths"),
        )

        # Req 6.1: the already-integrated commit (C1) is retained.
        shas = integrated_provenance_shas(self.fork_path, self.fork_base)
        self.assertIn(self.c1, shas)
        # Req 6.2: C3 (after the conflict) was NOT integrated while conflicted.
        self.assertNotIn(self.c3, shas)
        self.assertFalse(os.path.exists(os.path.join(self.fork_path, "feature2.txt")))
        # The tree is still mid-cherry-pick (conflicted), not advanced.
        self.assertTrue(
            os.path.exists(os.path.join(self.fork_path, ".git", "CHERRY_PICK_HEAD"))
        )

    def test_continuation_after_resolution_proceeds_in_order(self):
        proc = run_sync(self.fork_path, self.upstream_path, self.c0, self.c3)
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)

        # Manually resolve the conflict, then continue — the script's documented
        # recovery path (Req 6.5: continue in original upstream order).
        with open(os.path.join(self.fork_path, "common.txt"), "w", encoding="utf-8") as fh:
            fh.write("resolved\n")
        git(self.fork_path, "add", "common.txt")
        git(self.fork_path, "-c", "core.editor=true", "cherry-pick", "--continue")

        shas = integrated_provenance_shas(self.fork_path, self.fork_base)
        # Req 6.5: integration resumes in original upstream order C1 -> C2 -> C3.
        self.assertEqual(shas, [self.c1, self.c2, self.c3])
        self.assertTrue(os.path.exists(os.path.join(self.fork_path, "feature2.txt")))


class UsageTest(SyncUpstreamTestBase):
    def setUp(self) -> None:
        super().setUp()
        GitFixture.init(self.fork_path)

    def test_missing_arguments_is_usage_error(self):
        env = _full_env()
        env["REPO_PATH"] = self.fork_path
        proc = subprocess.run(
            ["bash", SCRIPT_PATH],
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 1)
        self.assertIn("usage", proc.stderr.lower())


if __name__ == "__main__":
    unittest.main()
