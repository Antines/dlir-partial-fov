#!/usr/bin/env python3
"""Produce a paired, common-case development table from rigid evaluation JSONL files."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from dl.rigid_dev_common import load_rows, summarize


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--inputs", required=True, nargs="+", type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("data_split") != "validation":
        raise ValueError("Only locked development validation samples are supported")
    case_ids = [c["relative_dir"] for c in manifest["cases"]]
    lookup = {}
    for path in args.inputs:
        for key, value in load_rows(path).items():
            if key in lookup:
                raise ValueError(f"Duplicate method/case entry in input files: {key}")
            lookup[key] = value
    methods = sorted(set(method for _, method in lookup))
    if not methods:
        raise ValueError("No recorded methods")
    records = [lookup.get((case, method), {"case": case, "method": method, "status": "missing", "failure": True})
               for case in case_ids for method in methods]
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "paired_cases.csv").open("w", newline="") as f:
        fields = ["case", "subject", "method", "status", "mean_tre_mm", "rot_deg", "trans_mm", "failure", "runtime_s"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)
    comparison = summarize(records)
    for candidate in methods:
        if candidate == "initial_header":
            continue
        paired = [(lookup[(case, "initial_header")], lookup[(case, candidate)])
                  for case in case_ids if (case, "initial_header") in lookup and (case, candidate) in lookup
                  and lookup[(case, "initial_header")].get("status") == "ok"
                  and lookup[(case, candidate)].get("status") == "ok"]
        if paired:
            differences = np.asarray([x[1]["mean_tre_mm"] - x[0]["mean_tre_mm"] for x in paired])
            comparison[candidate]["paired_n_vs_header"] = len(differences)
            comparison[candidate]["median_paired_tre_delta_mm_vs_header"] = float(np.median(differences))
            comparison[candidate]["paired_better_than_header_pct"] = float(np.mean(differences < 0) * 100)
    (args.output / "summary.json").write_text(json.dumps(comparison, indent=2) + "\n")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
