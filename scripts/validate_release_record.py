#!/usr/bin/env python3
"""Validate a ``release_record.json`` provenance file.

The release record is the single source of provenance truth for a
kineto-spyre release (see the design's Data Models section). This helper loads
a record and asserts the schema and invariants that the rest of the pipeline
relies on:

* all required top-level keys are present;
* the ``subcomponents`` map contains exactly the four expected keys
  (``libaiupti``, ``aiu_toolkit``, ``pytorch``, ``kineto_spyre``) and each maps
  to exactly one non-empty version string (Requirement 9.1: every subcomponent
  has exactly one recorded version identifier);
* the invariant ``new_last_sync_commit == pinned_kineto_commit`` holds.

Usage::

    python validate_release_record.py <path/to/release_record.json>

Exits ``0`` when the record is valid, non-zero with a clear message otherwise.
"""

from __future__ import annotations

import json
import sys
from typing import Any, List

# Top-level keys every release record must contain.
REQUIRED_TOP_LEVEL_KEYS = (
    "release",
    "target_pytorch",
    "pinned_kineto_commit",
    "previous_last_sync_commit",
    "new_last_sync_commit",
    "integrated_commits",
    "subcomponents",
)

# The subcomponents map must contain exactly these keys, no more, no fewer.
EXPECTED_SUBCOMPONENTS = ("libaiupti", "aiu_toolkit", "pytorch", "kineto_spyre")


class ReleaseRecordError(Exception):
    """Raised when a release record is malformed or violates the schema."""


def load_release_record(path: str) -> Any:
    """Load and parse a release_record.json file.

    Raises ``ReleaseRecordError`` with a clear message on a missing file or
    malformed JSON.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except FileNotFoundError as exc:
        raise ReleaseRecordError(f"release record not found: {path}") from exc
    except OSError as exc:
        raise ReleaseRecordError(f"could not read release record {path}: {exc}") from exc

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ReleaseRecordError(
            f"malformed JSON in {path}: {exc.msg} (line {exc.lineno}, column {exc.colno})"
        ) from exc


def validate_record(record: Any) -> None:
    """Validate a parsed release record against the schema and invariants.

    Raises ``ReleaseRecordError`` describing the first violation found.
    """
    if not isinstance(record, dict):
        raise ReleaseRecordError(
            f"release record must be a JSON object, got {type(record).__name__}"
        )

    # 1. All required top-level keys present.
    missing = [key for key in REQUIRED_TOP_LEVEL_KEYS if key not in record]
    if missing:
        raise ReleaseRecordError(
            "missing required top-level key(s): " + ", ".join(sorted(missing))
        )

    # 2. subcomponents map: exactly the four expected keys, each a non-empty version string.
    subcomponents = record["subcomponents"]
    if not isinstance(subcomponents, dict):
        raise ReleaseRecordError(
            "'subcomponents' must be a JSON object mapping subcomponent -> version"
        )

    actual_keys = set(subcomponents)
    expected_keys = set(EXPECTED_SUBCOMPONENTS)
    missing_subs = expected_keys - actual_keys
    extra_subs = actual_keys - expected_keys
    if missing_subs:
        raise ReleaseRecordError(
            "missing subcomponent(s): " + ", ".join(sorted(missing_subs))
        )
    if extra_subs:
        raise ReleaseRecordError(
            "unexpected subcomponent(s): " + ", ".join(sorted(extra_subs))
        )

    # Each subcomponent maps to exactly one non-empty version string (Req 9.1).
    for name in EXPECTED_SUBCOMPONENTS:
        version = subcomponents[name]
        if not isinstance(version, str):
            raise ReleaseRecordError(
                f"subcomponent '{name}' version must be a single string, "
                f"got {type(version).__name__}"
            )
        if version.strip() == "":
            raise ReleaseRecordError(
                f"subcomponent '{name}' has an empty version identifier; "
                "exactly one non-empty version is required (Req 9.1)"
            )

    # 3. Invariant: new_last_sync_commit == pinned_kineto_commit.
    pinned = record["pinned_kineto_commit"]
    new_sync = record["new_last_sync_commit"]
    if new_sync != pinned:
        raise ReleaseRecordError(
            "invariant violated: new_last_sync_commit "
            f"({new_sync!r}) must equal pinned_kineto_commit ({pinned!r})"
        )


def validate_release_record(path: str) -> None:
    """Load and validate a release record file. Raises on any problem."""
    record = load_release_record(path)
    validate_record(record)


def main(argv: List[str]) -> int:
    if len(argv) != 2:
        prog = argv[0] if argv else "validate_release_record.py"
        print(f"usage: {prog} <path/to/release_record.json>", file=sys.stderr)
        return 2

    path = argv[1]
    try:
        validate_release_record(path)
    except ReleaseRecordError as exc:
        print(f"ERROR: invalid release record: {exc}", file=sys.stderr)
        return 1

    print(f"OK: {path} is a valid release record")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
