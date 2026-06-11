"""Enable ``python -m tools.trace_validator <trace.json>``."""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
