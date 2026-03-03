"""Health check script for the Quantitative Fundamental agent dependencies."""

from __future__ import annotations

import json
import sys

from .tools import QuantFundamentalToolkit


def main() -> None:
    toolkit = QuantFundamentalToolkit()
    try:
        health = toolkit.healthcheck()
        print(json.dumps(health, indent=2))
        if not all(health.values()):
            sys.exit(1)
    finally:
        toolkit.close()


if __name__ == "__main__":
    main()
