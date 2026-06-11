#!/usr/bin/env python3
"""Tests for the release_record.json schema-validation helper.

Run from the repo root with::

    python3 -m unittest scripts.test_validate_release_record

or from this directory with::

    python3 -m unittest test_validate_release_record
"""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import unittest

# Make the helper importable whether run as `scripts.test_...` or directly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import validate_release_record as vrr  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_PATH = os.path.join(REPO_ROOT, "release_record.template.json")

# A 40-char SHA reused for pinned/new so the invariant holds.
SHA_A = "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678"
SHA_B = "0123456789abcdef0123456789abcdef01234567"

VALID_RECORD = {
    "release": "torch-2.12.0+aiu.kineto.1.2.0",
    "target_pytorch": "2.12.0",
    "pinned_kineto_commit": SHA_A,
    "previous_last_sync_commit": "7a731b6ae01cfc2b1fc75d83a91f84e682e43fd7",
    "new_last_sync_commit": SHA_A,
    "integrated_commits": [SHA_B, SHA_A],
    "subcomponents": {
        "libaiupti": "1.0.0",
        "aiu_toolkit": "2.3.1",
        "pytorch": "2.12.0",
        "kineto_spyre": "1.2.0",
    },
}


def write_temp(record_obj) -> str:
    """Write an object as JSON to a temp file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(record_obj, handle)
    return path


def write_temp_text(text: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


class ValidateRecordTest(unittest.TestCase):
    def test_valid_record_passes(self):
        # Should not raise.
        vrr.validate_record(copy.deepcopy(VALID_RECORD))

    def test_missing_top_level_key_fails(self):
        record = copy.deepcopy(VALID_RECORD)
        del record["target_pytorch"]
        with self.assertRaises(vrr.ReleaseRecordError) as ctx:
            vrr.validate_record(record)
        self.assertIn("target_pytorch", str(ctx.exception))

    def test_subcomponent_empty_version_fails(self):
        record = copy.deepcopy(VALID_RECORD)
        record["subcomponents"]["libaiupti"] = "   "
        with self.assertRaises(vrr.ReleaseRecordError) as ctx:
            vrr.validate_record(record)
        self.assertIn("libaiupti", str(ctx.exception))

    def test_subcomponent_missing_fails(self):
        record = copy.deepcopy(VALID_RECORD)
        del record["subcomponents"]["kineto_spyre"]
        with self.assertRaises(vrr.ReleaseRecordError) as ctx:
            vrr.validate_record(record)
        self.assertIn("kineto_spyre", str(ctx.exception))

    def test_subcomponent_extra_fails(self):
        record = copy.deepcopy(VALID_RECORD)
        record["subcomponents"]["unexpected_dep"] = "9.9.9"
        with self.assertRaises(vrr.ReleaseRecordError) as ctx:
            vrr.validate_record(record)
        self.assertIn("unexpected_dep", str(ctx.exception))

    def test_subcomponent_non_string_version_fails(self):
        record = copy.deepcopy(VALID_RECORD)
        record["subcomponents"]["pytorch"] = ["2.12.0"]
        with self.assertRaises(vrr.ReleaseRecordError) as ctx:
            vrr.validate_record(record)
        self.assertIn("pytorch", str(ctx.exception))

    def test_sync_commit_mismatch_fails(self):
        record = copy.deepcopy(VALID_RECORD)
        record["new_last_sync_commit"] = SHA_B
        with self.assertRaises(vrr.ReleaseRecordError) as ctx:
            vrr.validate_record(record)
        self.assertIn("new_last_sync_commit", str(ctx.exception))

    def test_non_object_record_fails(self):
        with self.assertRaises(vrr.ReleaseRecordError):
            vrr.validate_record(["not", "an", "object"])


class LoadRecordTest(unittest.TestCase):
    def test_malformed_json_reported(self):
        path = write_temp_text('{"release": "x",,}')
        try:
            with self.assertRaises(vrr.ReleaseRecordError) as ctx:
                vrr.load_release_record(path)
            self.assertIn("malformed JSON", str(ctx.exception))
        finally:
            os.remove(path)

    def test_missing_file_reported(self):
        with self.assertRaises(vrr.ReleaseRecordError) as ctx:
            vrr.load_release_record("/nonexistent/release_record.json")
        self.assertIn("not found", str(ctx.exception))

    def test_load_then_validate_valid_temp_file(self):
        path = write_temp(VALID_RECORD)
        try:
            vrr.validate_release_record(path)  # should not raise
        finally:
            os.remove(path)


class TemplateTest(unittest.TestCase):
    def test_template_is_valid(self):
        # The shipped template must itself pass validation.
        self.assertTrue(os.path.exists(TEMPLATE_PATH), TEMPLATE_PATH)
        vrr.validate_release_record(TEMPLATE_PATH)


class CliTest(unittest.TestCase):
    def test_cli_passes_on_valid(self):
        path = write_temp(VALID_RECORD)
        try:
            self.assertEqual(vrr.main(["validate_release_record.py", path]), 0)
        finally:
            os.remove(path)

    def test_cli_fails_on_invalid(self):
        record = copy.deepcopy(VALID_RECORD)
        del record["release"]
        path = write_temp(record)
        try:
            self.assertEqual(vrr.main(["validate_release_record.py", path]), 1)
        finally:
            os.remove(path)

    def test_cli_usage_error(self):
        self.assertEqual(vrr.main(["validate_release_record.py"]), 2)


if __name__ == "__main__":
    unittest.main()
