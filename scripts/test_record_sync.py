#!/usr/bin/env python3
"""Tests for the sync provenance scripts (design Component 3).

These drive the bash scripts (``record_sync.sh`` and
``check_lost_aiu_changes.sh``) via ``subprocess`` against throwaway fixture git
repositories created in temp dirs. Nothing here touches a live repo.

Run from the repo root with::

    python3 -m unittest scripts.test_record_sync

or from this directory with::

    python3 -m unittest test_record_sync
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RECORD_SYNC = os.path.join(SCRIPT_DIR, "record_sync.sh")
GUARD = os.path.join(SCRIPT_DIR, "check_lost_aiu_changes.sh")
VALIDATOR = os.path.join(SCRIPT_DIR, "validate_release_record.py")

# Fabricated 40-hex upstream SHAs (ascending), newest == pinned commit.
UPSTREAM_SHAS = [
    "1111111111111111111111111111111111111111",
    "2222222222222222222222222222222222222222",
    "b2103f78d13fde4937af010c0ef8e24313568bc5",  # PyTorch v2.12.0 pin (newest)
]

README_SYNC_LINE = (
    "The last upstream sync was with the commit "
    "`7a731b6ae01cfc2b1fc75d83a91f84e682e43fd7`."
)


def run(cmd, cwd=None, env=None):
    """Run a command, returning the CompletedProcess (captured output)."""
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=full_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def git(repo, *args):
    """Run a git command in *repo*, raising on failure."""
    proc = run(["git", "-C", repo, *args])
    if proc.returncode != 0:
        raise AssertionError(
            f"git {' '.join(args)} failed: {proc.stderr.strip() or proc.stdout.strip()}"
        )
    return proc.stdout.strip()


def init_repo(path):
    """Initialise a git repo with deterministic identity/branch."""
    git(path, "init", "-q")
    git(path, "config", "user.email", "test@example.com")
    git(path, "config", "user.name", "Test User")
    git(path, "config", "commit.gpgsign", "false")


def write(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


class RecordSyncTest(unittest.TestCase):
    def setUp(self):
        self.repo = tempfile.mkdtemp(prefix="record_sync_")
        self.addCleanup(shutil.rmtree, self.repo, ignore_errors=True)
        init_repo(self.repo)

        # README carrying the existing sync line.
        readme = os.path.join(self.repo, "README.md")
        write(readme, f"# kineto-spyre\n\n{README_SYNC_LINE}\n\nMore text.\n")
        git(self.repo, "add", "README.md")
        git(self.repo, "commit", "-q", "-m", "initial: README with sync line")

        # This base commit is the previous Last_Sync point for the range.
        self.last_sync = git(self.repo, "rev-parse", "HEAD")

        # Create one commit per upstream SHA, each carrying an `-x` provenance
        # line, in ascending order.
        for i, sha in enumerate(UPSTREAM_SHAS):
            fname = os.path.join(self.repo, f"file_{i}.txt")
            write(fname, f"content {i}\n")
            git(self.repo, "add", f"file_{i}.txt")
            msg = f"integrate change {i}\n\n(cherry picked from commit {sha})"
            git(self.repo, "commit", "-q", "-m", msg)

        self.pinned = UPSTREAM_SHAS[-1]

    def _run_record_sync(self):
        return run(
            [
                "bash",
                RECORD_SYNC,
                "--repo", self.repo,
                "--last-sync", self.last_sync,
                "--pinned-kineto-commit", self.pinned,
                "--target-pytorch", "2.12.0",
                "--libaiupti-version", "1.0.0",
                "--aiu-toolkit-version", "2.3.1",
                "--kineto-spyre-version", "1.2.0",
            ]
        )

    def test_record_and_readme_and_schema(self):
        proc = self._run_record_sync()
        self.assertEqual(
            proc.returncode, 0,
            f"record_sync.sh failed:\nSTDOUT:{proc.stdout}\nSTDERR:{proc.stderr}",
        )

        record_path = os.path.join(self.repo, "release_record.json")
        self.assertTrue(os.path.exists(record_path), "release_record.json not written")
        with open(record_path, encoding="utf-8") as handle:
            record = json.load(handle)

        # Recorded SHAs match and are ascending (Req 1.2).
        self.assertEqual(record["integrated_commits"], UPSTREAM_SHAS)
        self.assertEqual(
            record["integrated_commits"],
            sorted(record["integrated_commits"], key=UPSTREAM_SHAS.index),
        )
        self.assertEqual(record["integrated_commits"], list(UPSTREAM_SHAS))

        # new_last_sync_commit is the newest integrated commit == pinned (Req 1.3).
        self.assertEqual(record["new_last_sync_commit"], UPSTREAM_SHAS[-1])
        self.assertEqual(record["new_last_sync_commit"], self.pinned)
        self.assertEqual(record["previous_last_sync_commit"], self.last_sync)
        self.assertEqual(record["target_pytorch"], "2.12.0")

        # README sync line rewritten to the new commit (Req 1.5).
        with open(os.path.join(self.repo, "README.md"), encoding="utf-8") as handle:
            readme = handle.read()
        self.assertIn(
            f"The last upstream sync was with the commit `{self.pinned}`.",
            readme,
        )
        self.assertNotIn("7a731b6ae01cfc2b1fc75d83a91f84e682e43fd7", readme)
        # PyTorch 2.12 target reflected in the README.
        self.assertIn("This release targets PyTorch `2.12`.", readme)

        # Produced record passes the schema validator (exactly one non-empty
        # version per subcomponent, invariant new == pinned).
        vproc = run(["python3", VALIDATOR, record_path])
        self.assertEqual(
            vproc.returncode, 0,
            f"validator rejected record:\n{vproc.stdout}\n{vproc.stderr}",
        )

    def test_idempotent_readme_no_duplicate_target_line(self):
        self.assertEqual(self._run_record_sync().returncode, 0)
        self.assertEqual(self._run_record_sync().returncode, 0)
        with open(os.path.join(self.repo, "README.md"), encoding="utf-8") as handle:
            readme = handle.read()
        self.assertEqual(readme.count("This release targets PyTorch `2.12`."), 1)

    def test_missing_required_subcomponent_version_fails(self):
        proc = run(
            [
                "bash", RECORD_SYNC,
                "--repo", self.repo,
                "--last-sync", self.last_sync,
                "--pinned-kineto-commit", self.pinned,
                # libaiupti-version intentionally omitted
                "--aiu-toolkit-version", "2.3.1",
                "--kineto-spyre-version", "1.2.0",
            ]
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("libaiupti", proc.stderr)


class LostAiuGuardTest(unittest.TestCase):
    AIU_FILE = "libkineto/src/plugin/aiupti/AiuptiActivityApi.cpp"
    NORMAL_FILE = "libkineto/src/CuptiActivityApi.cpp"

    def setUp(self):
        self.repo = tempfile.mkdtemp(prefix="lost_aiu_")
        self.addCleanup(shutil.rmtree, self.repo, ignore_errors=True)
        init_repo(self.repo)

        # Base: AIU plugin file + a normal file both present.
        write(os.path.join(self.repo, self.AIU_FILE), "// AIU original\n")
        write(os.path.join(self.repo, self.NORMAL_FILE), "// normal v1\n")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "base: AIU plugin + normal file")

        # An "upstream" commit that modifies ONLY the normal file (it does not
        # touch any AIU path).
        write(os.path.join(self.repo, self.NORMAL_FILE), "// normal v2 from upstream\n")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "upstream: touch only the normal file")
        self.upstream_sha = git(self.repo, "rev-parse", "HEAD")

    def test_guard_flags_lost_aiu_change(self):
        # Simulate a bad conflict resolution: HEAD overwrites the AIU file even
        # though the upstream commit never touched it.
        write(os.path.join(self.repo, self.AIU_FILE), "// AIU CLOBBERED\n")
        write(os.path.join(self.repo, self.NORMAL_FILE), "// normal v2 from upstream\n")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "bad resolution: clobbered AIU file")

        proc = run(["bash", GUARD, self.upstream_sha, self.repo])
        self.assertNotEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn(self.AIU_FILE, proc.stderr)
        self.assertIn("lost AIU-specific change", proc.stderr)

    def test_guard_passes_when_aiu_preserved(self):
        # A good resolution: HEAD only mirrors the upstream change to the normal
        # file; the AIU file is untouched.
        write(os.path.join(self.repo, self.NORMAL_FILE), "// normal v3 local resolution\n")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "good resolution: only normal file")

        proc = run(["bash", GUARD, self.upstream_sha, self.repo])
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("no AIU-specific changes were lost", proc.stdout)

    def test_guard_ignores_aiu_change_that_upstream_also_makes(self):
        # If the upstream commit itself modifies the AIU file, a matching HEAD
        # change is a legitimate integration, not a lost change.
        upstream_aiu = git  # alias for readability
        # New upstream commit that DOES touch the AIU file.
        write(os.path.join(self.repo, self.AIU_FILE), "// AIU changed upstream\n")
        upstream_aiu(self.repo, "add", "-A")
        upstream_aiu(self.repo, "commit", "-q", "-m", "upstream: modify AIU file")
        upstream2 = git(self.repo, "rev-parse", "HEAD")

        # HEAD applies the same AIU modification.
        write(os.path.join(self.repo, self.AIU_FILE), "// AIU changed (integrated)\n")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "integrate AIU change")

        proc = run(["bash", GUARD, upstream2, self.repo])
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)


if __name__ == "__main__":
    unittest.main()
