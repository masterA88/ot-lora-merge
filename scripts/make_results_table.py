"""Aggregate results/*.csv into a single markdown table for the README. No GPU needed.

Usage:  python scripts/make_results_table.py --glob "results/*.csv"
"""
from __future__ import annotations

import argparse
import csv
import glob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/*.csv")
    args = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(args.glob)):
        with open(path) as f:
            for r in csv.DictReader(f):
                if "method" in r and "avg_norm_acc" in r:
                    rows.append((r["method"], float(r["avg_norm_acc"])))

    if not rows:
        print("no results yet — run experiments/m*/run.py first")
        return

    rows.sort(key=lambda x: -x[1])
    print("| Method | avg-norm-acc |")
    print("|---|---|")
    for name, acc in rows:
        print(f"| {name} | {acc*100:.2f}% |")


if __name__ == "__main__":
    main()
