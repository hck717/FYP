"""Health check script for Business Analyst agent dependencies."""

from __future__ import annotations

import json
import sys
from .tools import BusinessAnalystToolkit


def main() -> None:
    toolkit = BusinessAnalystToolkit()
    try:
        health = toolkit.healthcheck()
        print(json.dumps(health, indent=2))
        if not all(health.values()):
            sys.exit(1)
    finally:
        toolkit.close()


if __name__ == "__main__":
    main()
