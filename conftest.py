# conftest.py
"""
Root conftest.py — adds the repo root to sys.path so that
    from agents.business_analyst.agent import ...
    from agents.web_search.agent import ...
work correctly when running pytest or python scripts from /FYP.

This is the standard pytest pattern for monorepo-style projects
that don't have a pip-installable package.
"""
import sys
from pathlib import Path

# Insert repo root (/FYP) at the front of sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
