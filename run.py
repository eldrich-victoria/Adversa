"""
run.py — Convenience launcher for the Streamlit application.

Usage:
    python run.py
"""

import subprocess
import sys
from pathlib import Path

APP = Path(__file__).parent / "app" / "streamlit_app.py"


def main() -> None:
    print("Starting Adversarial Toxic Language Auditor…")
    print(f"App: {APP}")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(APP), "--server.headless", "false"],
        check=True,
    )


if __name__ == "__main__":
    main()
