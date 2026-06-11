"""Config check for the release-wheel smoke workflow (Requirement 8.3).

This test asserts that ``.github/workflows/release-wheel-smoke.yml``:

* triggers on the ``release/2.12`` branch (Req 8.3), so a change submitted to
  the release branch runs the smoke test automatically;
* defines a ``smoke-aiu`` job that targets the ``[self-hosted, aiu]`` runner
  labels (the real release path, Req 8.6); and
* pins Python 3.12 (the release wheel is ``cp312``).

PyYAML is not part of a stock CPython install and the workflow runs on
GitHub-hosted runners regardless, so this test does not take a hard dependency
on it: when ``yaml`` is importable the assertions run against the parsed
structure, otherwise it falls back to text/regex assertions against the raw
file. Either path runs on a plain interpreter.
"""

import os
import re
import unittest

WORKFLOW_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ".github",
    "workflows",
    "release-wheel-smoke.yml",
)

RELEASE_BRANCH = "release/2.12"

try:
    import yaml  # type: ignore

    _HAVE_YAML = True
except ImportError:  # pragma: no cover - depends on interpreter environment
    _HAVE_YAML = False


def _read_workflow_text():
    with open(WORKFLOW_PATH, "r", encoding="utf-8") as handle:
        return handle.read()


class WorkflowConfigStructuredTest(unittest.TestCase):
    """Assertions against the parsed YAML (only when PyYAML is available)."""

    @classmethod
    def setUpClass(cls):
        if not _HAVE_YAML:
            raise unittest.SkipTest("PyYAML not available; structured checks skipped")
        with open(WORKFLOW_PATH, "r", encoding="utf-8") as handle:
            cls.doc = yaml.safe_load(handle)

    def _on(self):
        # PyYAML parses the bare key ``on:`` as the boolean True, so the
        # trigger block may live under either ``"on"`` or ``True``.
        doc = self.doc
        if "on" in doc:
            return doc["on"]
        return doc.get(True)

    def test_triggers_on_release_branch(self):
        on = self._on()
        self.assertIsNotNone(on, "workflow defines no trigger block")
        branches = []
        for event in ("push", "pull_request"):
            spec = on.get(event) if isinstance(on, dict) else None
            if isinstance(spec, dict) and spec.get("branches"):
                branches.extend(spec["branches"])
        self.assertIn(
            RELEASE_BRANCH,
            branches,
            "workflow must trigger on the {} branch".format(RELEASE_BRANCH),
        )

    def test_smoke_job_targets_self_hosted_aiu(self):
        jobs = self.doc.get("jobs", {})
        self.assertIn("smoke-aiu", jobs, "missing smoke-aiu job")
        runs_on = jobs["smoke-aiu"].get("runs-on")
        self.assertEqual(
            sorted(runs_on),
            sorted(["self-hosted", "aiu"]),
            "smoke-aiu must target the [self-hosted, aiu] labels",
        )

    def test_python_312_pinned(self):
        text = _read_workflow_text()
        self.assertIn("3.12", text, "Python 3.12 must be pinned in the workflow")


class WorkflowConfigTextTest(unittest.TestCase):
    """Text/regex assertions that run on any interpreter (no PyYAML needed)."""

    @classmethod
    def setUpClass(cls):
        cls.text = _read_workflow_text()

    def test_triggers_on_release_branch(self):
        # Branch appears (quoted or bare) in a branches: list under push/pull_request.
        self.assertRegex(
            self.text,
            r"branches:\s*\[\s*['\"]?{}['\"]?\s*\]".format(re.escape(RELEASE_BRANCH)),
            "workflow must list the {} branch in a branches filter".format(
                RELEASE_BRANCH
            ),
        )

    def test_has_push_and_pull_request_triggers(self):
        self.assertRegex(self.text, r"(?m)^\s*push:")
        self.assertRegex(self.text, r"(?m)^\s*pull_request:")

    def test_smoke_job_targets_self_hosted_aiu(self):
        self.assertRegex(
            self.text,
            r"(?m)^\s*smoke-aiu:",
            "missing smoke-aiu job",
        )
        self.assertRegex(
            self.text,
            r"runs-on:\s*\[\s*self-hosted\s*,\s*aiu\s*\]",
            "smoke-aiu must target the [self-hosted, aiu] labels",
        )

    def test_python_312_pinned(self):
        self.assertRegex(
            self.text,
            r"python-version:\s*['\"]?3\.12['\"]?",
            "Python 3.12 must be pinned via setup-python",
        )


if __name__ == "__main__":
    unittest.main()
