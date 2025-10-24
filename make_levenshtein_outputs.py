#!/usr/bin/env python3
# make_levenshtein_outputs.py
# Usage:
#   python make_levenshtein_outputs.py --in csv/combined_ocr.csv
# Outputs:
#   csv/distance.csv, csv/score.csv, csv/summary.csv

import csv
import argparse

def levenshtein(a: str, b: str) -> int:
    """Classic dynamic programming Levenshtein distance (integer)."""
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    if len(a) > len(b):
        a, b = b, a
    previous = list(range(len(a) + 1))
    for j, bj in enumerate(b, start=1):
        current = [j]
        for i, ai in enumerate(a, start=1):
            ins = previous[i] + 1
            delete = current[i - 1] + 1
            sub = previous[i - 1] + (0 if ai == bj else 1)
            current.append(min(ins, delete, sub))
        previous = current
    return previous[-1]

def normalize(s: str) -> str:
    if s is None:
        return ""
    return s.strip().casefold()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="csv/combined_ocr.csv", help="Input CSV (default: combined_ocr.csv)")
    args = ap.parse_args()

    input_path = args.inp
    distance_path = "csv/distance.csv"
    score_path = "csv/score.csv"
    summary_path = "csv/summary.csv"

    with open(input_path, "r", encoding="utf-8", newline="") as f_in:
        reader = list(csv.reader(f_in))
        if not reader:
            print("Empty input.")
            return
        header = reader[0]
        data_rows = reader[1:]

    # Prepare headers for outputs
    header_distance = header
    header_score = header

    # For summary accumulation
    methods = header[2:]  # columns from 3rd onward
    dist_sum = {m: 0 for m in methods}
    score_sum = {m: 0.0 for m in methods}
    count = {m: 0 for m in methods}

    # Write distance.csv and score.csv
    with open(distance_path, "w", encoding="utf-8", newline="") as f_dist, \
         open(score_path, "w", encoding="utf-8", newline="") as f_score:

        w_dist = csv.writer(f_dist)
        w_score = csv.writer(f_score)
        w_dist.writerow(header_distance)
        w_score.writerow(header_score)

        for row in data_rows:
            if not row:
                continue
            if len(row) < 2:
                continue

            img = row[0]
            truth_raw = row[1]
            truth = normalize(truth_raw)

            out_row_dist = [img, truth_raw]
            out_row_score = [img, truth_raw]

            for idx, cand_raw in enumerate(row[2:], start=2):
                cand = normalize(cand_raw)
                d = levenshtein(truth, cand)
                norm = 1 - d / max(len(truth), len(cand), 1)
                out_row_dist.append(str(d))
                out_row_score.append(f"{norm:.3f}")

                m = header[idx]
                dist_sum[m] += d
                score_sum[m] += norm
                count[m] += 1

            w_dist.writerow(out_row_dist)
            w_score.writerow(out_row_score)

    # Write summary.csv
    with open(summary_path, "w", encoding="utf-8", newline="") as f_sum:
        w = csv.writer(f_sum)
        w.writerow(["method", "sum_distance", "avg_score"])
        for m in methods:
            n = max(count[m], 1)
            w.writerow([m, dist_sum[m], f"{score_sum[m]/n:.3f}"])

    print(f"✅ Created:\n  {distance_path}\n  {score_path}\n  {summary_path}")

if __name__ == "__main__":
    main()
