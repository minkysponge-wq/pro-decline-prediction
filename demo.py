"""One-command demo of the whole project (no retraining; uses saved models/results).

    python demo.py            # run everything straight through (~30 s)
    python demo.py --pause    # stop after each section; press Enter to continue (live demo)

Requires that `python -m prodecline all` has been run once on this machine
(it creates data/ and results/models/, which are not stored in git).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

STEPS = [
    ("1. Tests: leakage, labels, features, all models (expect 22 passed)",
     [PY, "-m", "pytest", "-o", "addopts=", "-q"]),
    ("2. AUROC of every model, both outcomes, both lab strategies",
     [PY, "-m", "prodecline", "summary"]),
    ("3. Accuracy: default 0.5 vs cut-off tuned on training data",
     [PY, "-m", "prodecline", "improve-summary"]),
    ("4. Full classification report: precision, recall, F1, confusion matrix",
     [PY, "docs/classification_report.py"]),
    ("5. Classifying individual test patients: DROP (decline of 10+ points)",
     [PY, "docs/check_patients.py", "--outcome", "drop", "--n", "10"]),
    ("6. Classifying individual test patients: THRESHOLD (score of 40 or lower)",
     [PY, "docs/check_patients.py", "--outcome", "threshold", "--n", "10"]),
    ("7. The ceiling: best accuracy ANY model could reach on this data",
     [PY, "docs/accuracy_ceiling.py"]),
]


def preflight() -> None:
    missing = [p for p in ("data/processed/features.csv",
                           "results/models/drop__knn__ensemble.joblib",
                           "results/improved/metrics.csv")
               if not (ROOT / p).exists()]
    if missing:
        print("Missing:", ", ".join(missing))
        print("Run `python -m prodecline all` (and `python -m prodecline improve`) once first.")
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pause", action="store_true", help="wait for Enter between sections")
    args = ap.parse_args()
    preflight()

    failed = []
    for i, (title, cmd) in enumerate(STEPS):
        print("\n" + "=" * 90 + f"\n{title}\n" + "=" * 90, flush=True)
        rc = subprocess.run(cmd, cwd=ROOT).returncode
        if rc != 0:
            failed.append(title)
            print(f"!! step failed (exit code {rc})")
        if args.pause and i < len(STEPS) - 1:
            input("\n[press Enter for the next section] ")

    print("\n" + "=" * 90)
    if failed:
        print("FINISHED WITH PROBLEMS in:\n  - " + "\n  - ".join(failed))
        sys.exit(1)
    print("ALL DEMO STEPS PASSED. For the interactive walkthrough, open notebooks/demo.ipynb")


if __name__ == "__main__":
    main()
