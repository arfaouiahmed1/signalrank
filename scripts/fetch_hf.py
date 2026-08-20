#!/usr/bin/env python3
"""HF dump fetch — delegates to fetch_kaggle.py unified logic."""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    # Delegate to shared fetcher (keeps CLI compat for Makefile)
    script = Path(__file__).parent / "fetch_kaggle.py"
    sys.exit(subprocess.call([sys.executable, str(script)] + sys.argv[1:]))
