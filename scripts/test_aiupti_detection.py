#!/usr/bin/env python3
"""Integration test for AIUPTI detection (Task 9.8) — [needs AIU hardware].

This test makes ``libaiupti`` reachable ONLY through ``LIBAIUPTI_INSTALL_DIR``
and asserts that the real CMake detection path (FindAIUToolkit.cmake) finds it
and that ``HAS_AIUPTI`` is compiled in, as evidenced by the ``AIU library
found:`` line landing in ``build.log`` and the build-script gate confirming it.

It is guarded to SKIP when AIU hardware / a real ``libaiupti`` is unavailable
(the common case on developer machines and this authoring environment). The
guard logic itself is exercised hardware-independently by
``test_build_guards.py``.

Run from the repo root with::

    python3 -m unittest scripts.test_aiupti_detection

Requirements covered: 4.1, 4.2, 4.5.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_LIB = os.path.join(THIS_DIR, "build_lib.sh")


def _find_libaiupti():
    """Locate a real libaiupti via LIBAIUPTI_INSTALL_DIR or LD_LIBRARY_PATH."""
    candidates = []
    install_dir = os.environ.get("LIBAIUPTI_INSTALL_DIR")
    if install_dir:
        for sub in ("lib", "lib64"):
            candidates.append(os.path.join(install_dir, sub))
    candidates.extend(
        p for p in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if p
    )
    for d in candidates:
        if not d or not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            if name.startswith("libaiupti") and (".so" in name or name.endswith(".a")):
                return os.path.join(d, name)
    return None


AIU_AVAILABLE = _find_libaiupti() is not None


@unittest.skipUnless(
    AIU_AVAILABLE,
    "AIU hardware / libaiupti not available (set LIBAIUPTI_INSTALL_DIR to a real "
    "libaiupti install to run this end-to-end detection test)",
)
class AiuptiDetectionIntegrationTest(unittest.TestCase):
    """Req 4.1, 4.2, 4.5: libaiupti reachable only via LIBAIUPTI_INSTALL_DIR."""

    def test_detection_via_install_dir_sets_has_aiupti(self):
        lib = _find_libaiupti()
        self.assertIsNotNone(lib)
        # Locate the FindAIUToolkit.cmake module shipped by the fork plugin.
        repo_root = os.path.dirname(THIS_DIR)
        find_module = os.path.join(
            repo_root, "libkineto", "src", "plugin", "aiupti", "FindAIUToolkit.cmake"
        )
        self.assertTrue(
            os.path.exists(find_module),
            f"expected FindAIUToolkit.cmake at {find_module}",
        )
        install_dir = os.path.dirname(os.path.dirname(lib))  # strip lib(64)/<file>

        with tempfile.TemporaryDirectory() as d:
            # Minimal CMake project that uses the plugin's find module with the
            # library reachable only through LIBAIUPTI_INSTALL_DIR (Req 4.1).
            module_dir = os.path.dirname(find_module)
            cmakelists = os.path.join(d, "CMakeLists.txt")
            with open(cmakelists, "w", encoding="utf-8") as fh:
                fh.write(
                    "cmake_minimum_required(VERSION 3.18)\n"
                    "project(aiupti_detect)\n"
                    f'list(APPEND CMAKE_MODULE_PATH "{module_dir}")\n'
                    f'set(LIBAIUPTI_INSTALL_DIR "{install_dir}")\n'
                    "find_package(AIUToolkit)\n"
                    "if(AIU_LIBRARY)\n"
                    '  message(STATUS "AIU library found: ${AIU_LIBRARY}")\n'
                    "endif()\n"
                )
            build_log = os.path.join(d, "build.log")
            env = dict(os.environ)
            env["LIBAIUPTI_INSTALL_DIR"] = install_dir
            with open(build_log, "w", encoding="utf-8") as logfh:
                proc = subprocess.run(
                    ["cmake", "-S", d, "-B", os.path.join(d, "out")],
                    stdout=logfh,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
            self.assertEqual(proc.returncode, 0, "cmake configure failed")
            # Req 4.5 / 4.2: the build-script gate confirms detection + HAS_AIUPTI.
            res = subprocess.run(
                ["bash", BUILD_LIB, "assert_aiupti_detected", build_log],
                capture_output=True,
                text=True,
            )
            self.assertEqual(res.returncode, 0, res.stderr)
            confirm = subprocess.run(
                ["bash", BUILD_LIB, "confirm_wheel_has_aiupti", build_log],
                capture_output=True,
                text=True,
            )
            self.assertEqual(confirm.returncode, 0, confirm.stderr)


if __name__ == "__main__":
    unittest.main()
