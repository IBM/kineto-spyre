"""Unit tests for the validator CLI wrapper (task 6.2, Req 7.8)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from tools.trace_validator import cli

GOOD = {
    "traceEvents": [
        {"name": "k", "ph": "X", "ts": 100, "dur": 10, "pid": 1, "tid": 7, "cat": "privateuse1"},
    ]
}

BAD = {
    "traceEvents": [
        {"name": "k", "ph": "X", "ts": 0, "dur": 10, "pid": 1, "tid": 7, "cat": "privateuse1"},
    ]
}


class CliTests(unittest.TestCase):
    def _write(self, obj_or_text):
        fd, path = tempfile.mkstemp(suffix=".json")
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            if isinstance(obj_or_text, str):
                fh.write(obj_or_text)
            else:
                json.dump(obj_or_text, fh)
        return path

    def test_valid_trace_exits_zero(self):
        self.assertEqual(cli.main([self._write(GOOD)]), cli.EXIT_OK)

    def test_invalid_trace_exits_nonzero(self):
        self.assertEqual(cli.main([self._write(BAD)]), cli.EXIT_VIOLATIONS)

    def test_malformed_file_exits_parse_error(self):
        self.assertEqual(cli.main([self._write("{ not json")]), cli.EXIT_PARSE_ERROR)

    def test_multiple_traces_worst_exit_code_wins(self):
        rc = cli.main([self._write(GOOD), self._write(BAD)])
        self.assertEqual(rc, cli.EXIT_VIOLATIONS)


if __name__ == "__main__":
    unittest.main()
