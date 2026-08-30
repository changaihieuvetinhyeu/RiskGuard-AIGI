#!/usr/bin/env python
"""Print a compact status summary for locked sources and mandatory data."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    registry = json.loads(Path("artifacts/source_registry.lock.json").read_text(encoding="utf-8"))
    print(f"project={registry['project']}")
    print(f"alpha={registry['risk_control']['alpha']}")
    print(f"delta={registry['risk_control']['delta']}")
    for source in registry["sources"]:
        print(f"{source['name']}: {source['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
