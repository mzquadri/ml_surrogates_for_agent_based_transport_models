"""
Run generate_phase3_figures.py analyses one at a time.
Avoids conda stdout Unicode crash by using subprocess.
"""

import os
import sys
import subprocess

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
FIGURES_DIR = os.path.join(REPO, "scripts", "figure_generation")
SCRIPT = os.path.join(FIGURES_DIR, "generate_phase3_figures.py")
PYTHON = sys.executable

env = os.environ.copy()
env["PYTHONUTF8"] = "1"

analyses = ["3.1", "3.2", "3.3", "3.4", "3.5", "3.6"]

for a in analyses:
    print(f"\n{'=' * 60}")
    print(f"Running analysis {a}...")
    print(f"{'=' * 60}")
    log_file = os.path.join(REPO, f"phase3_analysis_{a.replace('.', '_')}.log")
    with open(log_file, "w", encoding="utf-8") as lf:
        result = subprocess.run(
            [PYTHON, SCRIPT, a],
            cwd=FIGURES_DIR,
            env=env,
            stdout=lf,
            stderr=subprocess.STDOUT,
        )
    if result.returncode == 0:
        print(f"  Analysis {a}: SUCCESS (log: {log_file})")
    else:
        print(f"  Analysis {a}: FAILED (rc={result.returncode}, log: {log_file})")
        # Print last 20 lines of log
        with open(log_file, "r", encoding="utf-8", errors="replace") as lf:
            lines = lf.readlines()
            for line in lines[-20:]:
                print(f"    {line.rstrip()}")

print("\nAll analyses attempted.")
