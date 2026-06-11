#!/usr/bin/env python3
"""Example tests for scripts/release_package.sh (Component 7, Requirement 10).

These drive the bash script via subprocess against temporary FIXTURE git repos
(a working repo plus a local *bare* repo acting as the 'remote'). Nothing here
touches a real remote or invokes the real GitHub CLI: pushes go to the local
bare repo and publishing stays in DRY_RUN mode.

Covered guards (Req 10.1, 10.2, 10.5, 10.6):
  * a tag whose name identifies 2.12 is created and contains "2.12";
  * exactly one wheel in the artifact set (0 or 2 wheels -> fail);
  * existing-tag conflict refusal (pre-create the tag -> conflict, non-zero,
    tag not moved/overwritten);
  * tag-creation failure publishes nothing (no artifacts staged/published).

stdlib unittest only; runs under Python 3.9 and 3.12.
"""

import json
import os
import shutil
import subprocess
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "release_package.sh")

PINNED_COMMIT = "b2103f78d13fde4937af010c0ef8e24313568bc5"
INTEGRATED = [
    "1111111111111111111111111111111111111111",
    "2222222222222222222222222222222222222222",
    PINNED_COMMIT,
]
KINETO_VERSION = "1.2.0"
EXPECTED_TAG = "torch-2.12.0.aiu.kineto.1.2.0"


def git(repo, *args, check=True):
    return subprocess.run(
        ["git", "-C", repo, *args],
        check=check,
        capture_output=True,
        text=True,
    )


class ReleasePackageTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="relpkg_")
        self.repo = os.path.join(self.tmp, "work")
        self.remote = os.path.join(self.tmp, "remote.git")
        os.makedirs(self.repo)

        # Working fixture repo.
        git(self.repo, "init", "-q", "-b", "main")
        git(self.repo, "config", "user.email", "test@example.com")
        git(self.repo, "config", "user.name", "Test")

        # README updated with the synced upstream commit ID + PyTorch 2.12 (Req 10.3).
        readme = os.path.join(self.repo, "README.md")
        with open(readme, "w") as fh:
            fh.write(
                "# kineto-spyre\n\n"
                "The last upstream sync was with the commit `%s`.\n\n"
                "Built against PyTorch 2.12.0.\n" % PINNED_COMMIT
            )

        # release_record.json with the integrated upstream commit IDs (Req 10.4).
        record = os.path.join(self.repo, "release_record.json")
        with open(record, "w") as fh:
            json.dump(
                {
                    "release": "torch-2.12.0+aiu.kineto.%s" % KINETO_VERSION,
                    "target_pytorch": "2.12.0",
                    "pinned_kineto_commit": PINNED_COMMIT,
                    "previous_last_sync_commit": "7a731b6ae01cfc2b1fc75d83a91f84e682e43fd7",
                    "new_last_sync_commit": PINNED_COMMIT,
                    "integrated_commits": INTEGRATED,
                    "subcomponents": {
                        "libaiupti": "0.9.0",
                        "aiu_toolkit": "1.0.0",
                        "pytorch": "2.12.0",
                        "kineto_spyre": KINETO_VERSION,
                    },
                },
                fh,
                indent=2,
            )

        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "fixture commit")

        # Bare repo acting as the 'remote' (never a real network remote).
        git(self.tmp, "init", "-q", "--bare", self.remote)

        # Wheel dir + staging dir.
        self.wheel_dir = os.path.join(self.tmp, "dist")
        os.makedirs(self.wheel_dir)
        self.staging = os.path.join(self.tmp, "staging")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def make_wheel(self, name):
        path = os.path.join(self.wheel_dir, name)
        with open(path, "wb") as fh:
            fh.write(b"PK\x03\x04 fake wheel")
        return path

    def run_script(self, extra_env=None, smoke_passed="1"):
        env = dict(os.environ)
        env.update(
            {
                "REPO_PATH": self.repo,
                "REMOTE": self.remote,
                "WHEEL_DIR": self.wheel_dir,
                "STAGING_DIR": self.staging,
                "RELEASE_RECORD": os.path.join(self.repo, "release_record.json"),
                "README_PATH": os.path.join(self.repo, "README.md"),
                "KINETO_VERSION": KINETO_VERSION,
                "PYTORCH_VERSION": "2.12.0",
                "SMOKE_PASSED": smoke_passed,
                "DRY_RUN": "1",  # never invoke real gh / network
            }
        )
        if extra_env:
            env.update(extra_env)
        return subprocess.run(
            ["bash", SCRIPT],
            env=env,
            capture_output=True,
            text=True,
        )

    def staged_wheels(self):
        if not os.path.isdir(self.staging):
            return []
        return [f for f in os.listdir(self.staging) if f.endswith(".whl")]


class TestHappyPath(ReleasePackageTestBase):
    def test_creates_one_2_12_tag_and_assembles_single_wheel_artifact_set(self):
        """Req 10.1/10.2: one 2.12 tag created; exactly one wheel in the set."""
        self.make_wheel(
            "torch-2.12.0+aiu.kineto.1.2.0-cp312-cp312-linux_x86_64.whl"
        )
        res = self.run_script()
        self.assertEqual(res.returncode, 0, msg=res.stderr)

        # Exactly one tag, named to identify 2.12, exists (Req 10.1).
        tags = git(self.repo, "tag", "--list").stdout.split()
        self.assertEqual(tags, [EXPECTED_TAG])
        self.assertIn("2.12", EXPECTED_TAG)

        # Annotated tag.
        kind = git(self.repo, "cat-file", "-t", EXPECTED_TAG).stdout.strip()
        self.assertEqual(kind, "tag")

        # Tag pushed to the local bare 'remote'.
        remote_tags = git(self.remote, "tag", "--list").stdout.split()
        self.assertIn(EXPECTED_TAG, remote_tags)

        # Exactly one wheel staged in the artifact set (Req 10.2).
        self.assertEqual(len(self.staged_wheels()), 1)

        # README + release_record + notes assembled (Req 10.3/10.4).
        for fname in ("README.md", "release_record.json", "RELEASE_NOTES.md"):
            self.assertTrue(
                os.path.isfile(os.path.join(self.staging, fname)),
                msg="missing %s" % fname,
            )

        # Notes associate every integrated upstream commit ID with the tag (Req 10.4).
        with open(os.path.join(self.staging, "RELEASE_NOTES.md")) as fh:
            notes = fh.read()
        for sha in INTEGRATED:
            self.assertIn(sha, notes)


class TestWheelCount(ReleasePackageTestBase):
    def test_zero_wheels_fails_before_tagging(self):
        """Req 10.2: 0 wheels -> fail, no tag, nothing published."""
        res = self.run_script()
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("exactly 1 wheel", res.stderr)
        self.assertEqual(git(self.repo, "tag", "--list").stdout.strip(), "")
        self.assertEqual(self.staged_wheels(), [])

    def test_two_wheels_fails_before_tagging(self):
        """Req 10.2: 2 wheels -> fail, no tag, nothing published."""
        self.make_wheel("torch-2.12.0+aiu.kineto.1.2.0-cp312-cp312-linux_x86_64.whl")
        self.make_wheel("torch-2.12.0+aiu.kineto.1.2.0-cp39-cp39-linux_x86_64.whl")
        res = self.run_script()
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("exactly 1 wheel", res.stderr)
        self.assertEqual(git(self.repo, "tag", "--list").stdout.strip(), "")
        self.assertFalse(os.path.isdir(self.staging))


class TestExistingTagConflict(ReleasePackageTestBase):
    def test_refuses_to_overwrite_existing_tag(self):
        """Req 10.5: pre-existing tag -> conflict, non-zero, not overwritten."""
        self.make_wheel("torch-2.12.0+aiu.kineto.1.2.0-cp312-cp312-linux_x86_64.whl")

        # Pre-create the tag pointing at an initial commit.
        git(self.repo, "tag", "-a", EXPECTED_TAG, "-m", "pre-existing")
        original = git(self.repo, "rev-list", "-n", "1", EXPECTED_TAG).stdout.strip()

        # Add a new commit so an overwrite (if it happened) would move the tag.
        with open(os.path.join(self.repo, "extra.txt"), "w") as fh:
            fh.write("change\n")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "later commit")

        res = self.run_script()
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("conflict", res.stderr.lower())

        # Tag still points at the ORIGINAL commit (not overwritten, Req 10.5).
        after = git(self.repo, "rev-list", "-n", "1", EXPECTED_TAG).stdout.strip()
        self.assertEqual(after, original)

        # Nothing published/staged.
        self.assertFalse(os.path.isdir(self.staging))


class TestTagCreationFailurePublishesNothing(ReleasePackageTestBase):
    def test_tag_creation_failure_publishes_nothing(self):
        """Req 10.6: a tag-creation failure publishes nothing.

        Force a failure with an invalid tag name (contains spaces); git refuses
        to create it. The script must stop and stage/publish no artifacts.
        """
        self.make_wheel("torch-2.12.0+aiu.kineto.1.2.0-cp312-cp312-linux_x86_64.whl")
        res = self.run_script(extra_env={"TAG": "torch 2.12 invalid name"})
        self.assertNotEqual(res.returncode, 0)

        # No tag created at all.
        self.assertEqual(git(self.repo, "tag", "--list").stdout.strip(), "")
        # Nothing pushed to the remote.
        self.assertEqual(git(self.remote, "tag", "--list").stdout.strip(), "")
        # No artifact set staged.
        self.assertEqual(self.staged_wheels(), [])


class TestSmokePrecondition(ReleasePackageTestBase):
    def test_refuses_when_smoke_not_passed(self):
        """Req 10.1: tag created only after the smoke test passed."""
        self.make_wheel("torch-2.12.0+aiu.kineto.1.2.0-cp312-cp312-linux_x86_64.whl")
        res = self.run_script(smoke_passed="0")
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("smoke", res.stderr.lower())
        self.assertEqual(git(self.repo, "tag", "--list").stdout.strip(), "")
        self.assertFalse(os.path.isdir(self.staging))


if __name__ == "__main__":
    unittest.main()
