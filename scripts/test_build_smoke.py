#!/usr/bin/env python3
"""Smoke / config checks for the build script (Task 9.9).

These are single-execution static/smoke assertions over ``build_pytorch.sh``
and ``build_lib.sh`` — they do not run a real build:

* the build version targets 2.12 (Req 2.3, 5.2);
* PyTorch's own ``setup.py build`` / ``bdist_wheel`` interface is invoked
  unmodified — the script never edits PyTorch build files (Req 5.5);
* the build path emits a detected-or-not AIUPTI line into ``build.log`` (Req 4.3).

Run from the repo root with::

    python3 -m unittest scripts.test_build_smoke
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import unittest

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_LIB = os.path.join(THIS_DIR, "build_lib.sh")
BUILD_SCRIPT = os.path.join(THIS_DIR, "build_pytorch.sh")


def read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


class BuildVersionSmokeTest(unittest.TestCase):
    """Req 2.3 / 5.2: build version is 2.12."""

    def test_pytorch_version_pinned_to_212(self):
        script = read(BUILD_SCRIPT)
        self.assertRegex(script, r'PYTORCH_VERSION="2\.12\.0"')

    def test_python_abi_is_cp312(self):
        # The release ships a cp312 wheel; the script pins the build Python to 3.12.
        script = read(BUILD_SCRIPT)
        self.assertRegex(script, r'PYTHON_RELEASE_VERSION="\$\{PYTHON_RELEASE_VERSION:-3\.12\}"')

    def test_build_version_string_targets_212(self):
        # PYTORCH_BUILD_VERSION = PYTORCH_VERSION + suffix → starts with 2.12.
        res = subprocess.run(
            ["bash", BUILD_LIB, "assert_build_version", "2.12.0+aiu.kineto.1.2.0"],
            capture_output=True, text=True,
        )
        self.assertEqual(res.returncode, 0, res.stderr)


class UnmodifiedInterfaceSmokeTest(unittest.TestCase):
    """Req 5.5: PyTorch's setup.py build/bdist_wheel interface invoked unmodified."""

    def test_invokes_setup_py_build_and_bdist_wheel(self):
        script = read(BUILD_SCRIPT)
        self.assertIn("setup.py build", script)
        self.assertIn("setup.py bdist_wheel", script)

    def test_does_not_edit_pytorch_build_files(self):
        """The script only sets env + swaps the submodule; it never edits
        PyTorch's own build files (setup.py / CMakeLists / build_variables)."""
        script = read(BUILD_SCRIPT)
        # No in-place edits targeting PyTorch build files.
        forbidden = [
            r"sed\s+-i[^\n]*setup\.py",
            r"sed\s+-i[^\n]*CMakeLists",
            r">\s*\S*setup\.py",          # redirect-overwrite of setup.py
            r"patch[^\n]*setup\.py",
        ]
        for pat in forbidden:
            self.assertIsNone(
                re.search(pat, script),
                f"build script must not modify PyTorch build files (matched {pat})",
            )


class AiuptiLogLineSmokeTest(unittest.TestCase):
    """Req 4.3: build.log carries the detected-or-not AIUPTI line."""

    def test_detected_line_written(self):
        with tempfile.TemporaryDirectory() as d:
            log = os.path.join(d, "build.log")
            with open(log, "w", encoding="utf-8") as fh:
                fh.write("-- AIU library found: /opt/aiu/lib/libaiupti.so\n")
            subprocess.run(["bash", BUILD_LIB, "assert_aiupti_detected", log],
                           capture_output=True, text=True)
            self.assertIn("AIUPTI detected:", read(log))

    def test_not_detected_line_written(self):
        with tempfile.TemporaryDirectory() as d:
            log = os.path.join(d, "build.log")
            with open(log, "w", encoding="utf-8") as fh:
                fh.write("AIU PTI has not built\n")
            subprocess.run(["bash", BUILD_LIB, "assert_aiupti_detected", log],
                           capture_output=True, text=True)
            self.assertIn("not detected", read(log))

    def test_build_script_tees_to_build_log(self):
        # The build step must tee output to build.log so the gate can scan it.
        script = read(BUILD_SCRIPT)
        self.assertIn("tee build.log", script)
        self.assertIn("assert_aiupti_detected", script)


if __name__ == "__main__":
    unittest.main()
