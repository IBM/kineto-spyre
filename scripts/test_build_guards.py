#!/usr/bin/env python3
"""Example/edge-case tests for the build-script guards (Task 9.7).

These tests drive the individually-callable guard functions in
``scripts/build_lib.sh`` against small temp fixtures, so the gate logic is
verified WITHOUT cloning PyTorch or running a real (multi-GB, AIU-hardware)
build. Each guard is invoked through the dispatcher::

    bash scripts/build_lib.sh <function> [args...]

Run from the repo root with::

    python3 -m unittest scripts.test_build_guards

or from this directory with::

    python3 -m unittest test_build_guards

Requirements covered: 2.7, 3.2, 4.4, 4.6, 5.3, 5.4, 5.6, 5.7, 5.8, 9.3, 9.4, 9.5.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
BUILD_LIB = os.path.join(THIS_DIR, "build_lib.sh")
BUILD_SCRIPT = os.path.join(THIS_DIR, "build_pytorch.sh")

# Exit codes the helper documents.
EXIT_AIUPTI_MISSING = 3


def run_guard(func, *args, env_extra=None):
    """Invoke a build_lib.sh function via the dispatcher; return CompletedProcess."""
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        ["bash", BUILD_LIB, func, *args],
        capture_output=True,
        text=True,
        env=env,
    )


def write(path, text):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


class VersionGuardTest(unittest.TestCase):
    """Req 2.7 / 5.7: source not 2.12 → report mismatch + stop."""

    def test_version_212_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            vf = os.path.join(d, "version.txt")
            write(vf, "2.12.0a0\n")
            res = run_guard("verify_pytorch_version", vf)
            self.assertEqual(res.returncode, 0, res.stderr)

    def test_version_212_release_accepted(self):
        with tempfile.TemporaryDirectory() as d:
            vf = os.path.join(d, "version.txt")
            write(vf, "2.12.0\n")
            res = run_guard("verify_pytorch_version", vf)
            self.assertEqual(res.returncode, 0, res.stderr)

    def test_non_212_stops(self):
        with tempfile.TemporaryDirectory() as d:
            vf = os.path.join(d, "version.txt")
            write(vf, "2.11.0\n")
            res = run_guard("verify_pytorch_version", vf)
            self.assertNotEqual(res.returncode, 0)
            self.assertIn("2.11.0", res.stderr)
            self.assertIn("2.12", res.stderr)

    def test_missing_version_file_stops(self):
        with tempfile.TemporaryDirectory() as d:
            res = run_guard("verify_pytorch_version", os.path.join(d, "nope.txt"))
            self.assertNotEqual(res.returncode, 0)

    def test_build_version_must_target_212(self):
        ok = run_guard("assert_build_version", "2.12.0+aiu.kineto.1.2.0")
        self.assertEqual(ok.returncode, 0, ok.stderr)
        bad = run_guard("assert_build_version", "2.11.0+aiu.kineto.1.1.2")
        self.assertNotEqual(bad.returncode, 0)


class KinetoReplacementTest(unittest.TestCase):
    """Req 5.3 / 5.8: incomplete kineto replacement → stop."""

    def test_complete_replacement_ok(self):
        with tempfile.TemporaryDirectory() as d:
            plugin = os.path.join(
                d, "third_party", "kineto", "libkineto", "src", "plugin", "aiupti"
            )
            os.makedirs(plugin)
            res = run_guard("verify_kineto_replacement", d)
            self.assertEqual(res.returncode, 0, res.stderr)

    def test_missing_aiu_plugin_stops(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "third_party", "kineto", "libkineto", "src"))
            res = run_guard("verify_kineto_replacement", d)
            self.assertNotEqual(res.returncode, 0)
            self.assertIn("AIU plugin", res.stderr)


class PrivateUse1Test(unittest.TestCase):
    """Req 3.2: missing PrivateUse1 → halt before wheel build + log entry."""

    def _make_src(self, d, contents):
        autograd = os.path.join(d, "torch", "csrc", "autograd")
        os.makedirs(autograd)
        write(os.path.join(autograd, "init.cpp"), contents)

    def test_macro_present_ok(self):
        with tempfile.TemporaryDirectory() as d:
            self._make_src(d, "REGISTER_PRIVATEUSE1_PROFILER(foo)\n")
            log = os.path.join(d, "build.log")
            res = run_guard("check_privateuse1", d, log)
            self.assertEqual(res.returncode, 0, res.stderr)

    def test_api_present_ok(self):
        with tempfile.TemporaryDirectory() as d:
            self._make_src(d, "void x(){ registerPrivateUse1Activity(); }\n")
            log = os.path.join(d, "build.log")
            res = run_guard("check_privateuse1", d, log)
            self.assertEqual(res.returncode, 0, res.stderr)

    def test_missing_stops_and_logs(self):
        with tempfile.TemporaryDirectory() as d:
            self._make_src(d, "// nothing relevant here\n")
            log = os.path.join(d, "build.log")
            res = run_guard("check_privateuse1", d, log)
            self.assertNotEqual(res.returncode, 0)
            # Req 3.2: build-log entry names the missing dependency.
            self.assertTrue(os.path.exists(log))
            with open(log, encoding="utf-8") as fh:
                log_text = fh.read()
            self.assertIn("PrivateUse1_Registration", log_text)
            # No wheel is produced because the function returns non-zero before
            # the build is invoked (verified by the build-script ordering test).


class AiuptiGateTest(unittest.TestCase):
    """Req 4.4 / 4.6: libaiupti not detected → abort, no override, no wheel."""

    def _log(self, d, text):
        p = os.path.join(d, "build.log")
        write(p, text)
        return p

    def test_detected_passes_and_logs_path(self):
        with tempfile.TemporaryDirectory() as d:
            log = self._log(d, "cmake...\n-- AIU library found: /opt/aiu/lib/libaiupti.so\n")
            res = run_guard("assert_aiupti_detected", log)
            self.assertEqual(res.returncode, 0, res.stderr)
            with open(log, encoding="utf-8") as fh:
                text = fh.read()
            self.assertIn("AIUPTI detected:", text)
            self.assertIn("/opt/aiu/lib/libaiupti.so", text)

    def test_not_detected_aborts_nonzero(self):
        with tempfile.TemporaryDirectory() as d:
            log = self._log(d, "cmake...\nAIU PTI has not built\n")
            res = run_guard("assert_aiupti_detected", log)
            # Req 4.4: non-zero detection-failure status.
            self.assertEqual(res.returncode, EXIT_AIUPTI_MISSING)
            with open(log, encoding="utf-8") as fh:
                text = fh.read()
            # Req 4.3: a not-detected line is emitted to the log.
            self.assertIn("not detected", text)

    def test_gate_is_unconditional_no_override(self):
        """Req 4.6: no build-type / override argument relaxes the gate.

        Passing extra args or common 'override' env vars must NOT turn the
        abort into a pass — the gate only looks at the build log.
        """
        with tempfile.TemporaryDirectory() as d:
            log = self._log(d, "AIU PTI has not built\n")
            for env_extra in (
                {"ALLOW_NO_AIUPTI": "1"},
                {"LIBKINETO_NOAIUPTI": "1"},
                {"BUILD_TYPE": "debug"},
                {"BUILD_TYPE": "release"},
            ):
                res = run_guard("assert_aiupti_detected", log, env_extra=env_extra)
                self.assertEqual(
                    res.returncode, EXIT_AIUPTI_MISSING,
                    f"override env {env_extra} must not relax the gate",
                )

    def test_build_script_has_no_override_path(self):
        """Req 4.6: the build script exposes no override flag / warn-continue."""
        with open(BUILD_SCRIPT, encoding="utf-8") as fh:
            script = fh.read()
        with open(BUILD_LIB, encoding="utf-8") as fh:
            lib = fh.read()
        combined = script + lib
        for forbidden in ("--allow-no-aiupti", "ALLOW_NO_AIUPTI", "LIBKINETO_NOAIUPTI=1"):
            self.assertNotIn(
                forbidden, combined,
                f"build path must not contain an AIUPTI override: {forbidden}",
            )

    def test_confirm_requires_detection_line(self):
        with tempfile.TemporaryDirectory() as d:
            good = self._log(d, "AIUPTI detected: /opt/aiu/lib/libaiupti.so\n")
            self.assertEqual(run_guard("confirm_wheel_has_aiupti", good).returncode, 0)
        with tempfile.TemporaryDirectory() as d:
            bad = self._log(d, "no detection here\n")
            self.assertNotEqual(run_guard("confirm_wheel_has_aiupti", bad).returncode, 0)


class WheelPostconditionTest(unittest.TestCase):
    """Req 5.4 / 5.6: exactly one wheel; 0 or 2 → fail + remove partials."""

    def test_single_wheel_ok(self):
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "torch-2.12.0.whl"), "w").close()
            self.assertEqual(run_guard("assert_single_wheel", d).returncode, 0)

    def test_zero_wheels_fail(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertNotEqual(run_guard("assert_single_wheel", d).returncode, 0)

    def test_two_wheels_fail_and_cleanup(self):
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "a.whl"), "w").close()
            open(os.path.join(d, "b.whl"), "w").close()
            res = run_guard("assert_single_wheel", d)
            self.assertNotEqual(res.returncode, 0)
            # Req 5.6: leave no partial wheel.
            remaining = [f for f in os.listdir(d) if f.endswith(".whl")]
            self.assertEqual(remaining, [])

    def test_build_failure_cleanup(self):
        """Req 5.6: report failing stage and remove any partial wheel."""
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "partial.whl"), "w").close()
            res = run_guard("report_build_failure", "setup.py build", d)
            self.assertNotEqual(res.returncode, 0)
            self.assertIn("setup.py build", res.stderr)
            remaining = [f for f in os.listdir(d) if f.endswith(".whl")]
            self.assertEqual(remaining, [])

    def test_clean_dist_clears(self):
        with tempfile.TemporaryDirectory() as d:
            dist = os.path.join(d, "dist")
            os.makedirs(dist)
            open(os.path.join(dist, "stale.whl"), "w").close()
            res = run_guard("clean_dist", dist)
            self.assertEqual(res.returncode, 0, res.stderr)
            self.assertTrue(os.path.isdir(dist))
            self.assertEqual(os.listdir(dist), [])


class SubcomponentVersionTest(unittest.TestCase):
    """Req 9.3 / 9.4 / 9.5: missing / unobtainable / mismatched versions → stop."""

    RECORD = {
        "subcomponents": {
            "libaiupti": "1.0.0",
            "aiu_toolkit": "2.3.1",
            "pytorch": "2.12.0",
            "kineto_spyre": "1.2.0",
        }
    }

    def _files(self, d, record, obtained):
        rec = os.path.join(d, "release_record.json")
        obt = os.path.join(d, "obtained.json")
        write(rec, json.dumps(record))
        write(obt, json.dumps({"subcomponents": obtained}))
        return rec, obt

    def test_all_match_ok(self):
        with tempfile.TemporaryDirectory() as d:
            rec, obt = self._files(d, self.RECORD, self.RECORD["subcomponents"])
            res = run_guard(
                "verify_subcomponents", rec, env_extra={"OBTAINED_VERSIONS_JSON": obt}
            )
            self.assertEqual(res.returncode, 0, res.stderr)

    def test_missing_recorded_version_stops(self):
        rec_obj = json.loads(json.dumps(self.RECORD))
        rec_obj["subcomponents"]["libaiupti"] = ""
        with tempfile.TemporaryDirectory() as d:
            rec, obt = self._files(d, rec_obj, self.RECORD["subcomponents"])
            res = run_guard(
                "verify_subcomponents", rec, env_extra={"OBTAINED_VERSIONS_JSON": obt}
            )
            self.assertNotEqual(res.returncode, 0)
            self.assertIn("no recorded version", res.stderr)
            self.assertIn("libaiupti", res.stderr)

    def test_unobtainable_version_stops(self):
        obtained = dict(self.RECORD["subcomponents"])
        del obtained["pytorch"]  # cannot be obtained
        with tempfile.TemporaryDirectory() as d:
            rec, obt = self._files(d, self.RECORD, obtained)
            res = run_guard(
                "verify_subcomponents", rec, env_extra={"OBTAINED_VERSIONS_JSON": obt}
            )
            self.assertNotEqual(res.returncode, 0)
            self.assertIn("cannot be obtained", res.stderr)
            self.assertIn("pytorch", res.stderr)

    def test_mismatch_names_subcomponent_and_both_versions(self):
        obtained = dict(self.RECORD["subcomponents"])
        obtained["libaiupti"] = "9.9.9"
        with tempfile.TemporaryDirectory() as d:
            rec, obt = self._files(d, self.RECORD, obtained)
            res = run_guard(
                "verify_subcomponents", rec, env_extra={"OBTAINED_VERSIONS_JSON": obt}
            )
            self.assertNotEqual(res.returncode, 0)
            self.assertIn("libaiupti", res.stderr)   # subcomponent
            self.assertIn("1.0.0", res.stderr)        # recorded
            self.assertIn("9.9.9", res.stderr)        # obtained


if __name__ == "__main__":
    unittest.main()
