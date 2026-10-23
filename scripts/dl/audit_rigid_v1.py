#!/usr/bin/env python3
"""Paired audit of the frozen 1,080-case rigid validation records."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def read_jsonl(path: Path, method: str) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get('method') != method:
            continue
        case = record['case']
        if case in records and record != records[case]:
            raise ValueError(f'Conflicting duplicate {method}: {case}')
        records[case] = record
    if not records:
        raise ValueError(f'No records for {method}: {path}')
    return records


def robust_median(values):
    return float(np.median(values)) if values else None


def percentile(values, q):
    return float(np.percentile(values, q)) if values else None


def group_stats(rows: list[dict], key: str) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    out = []
    for label, subset in sorted(groups.items()):
        refined = [r['refined_tre_mm'] for r in subset]
        out.append({
            'group': key, 'value': label, 'n': len(subset),
            'median_refined_tre_mm': robust_median(refined),
            'p90_refined_tre_mm': percentile(refined, 90),
            'failure_pct': 100 * np.mean([r['failure'] for r in subset]),
            'tre_below_2mm_pct': 100 * np.mean([x < 2 for x in refined]),
            'improved_vs_header_pct': 100 * np.mean([r['improved'] for r in subset]),
        })
    return out


def audit(manifest: dict, header: dict[str, dict], refined: dict[str, dict]) -> tuple[dict, list[dict], list[dict]]:
    if manifest.get('data_split') != 'validation':
        raise ValueError('Expected locked validation manifest')
    cases = manifest['cases']
    ids = [c['relative_dir'] for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate case IDs in manifest')
    if set(header) != set(ids) or set(refined) != set(ids):
        missing = set(ids) - set(header) | (set(ids) - set(refined))
        extra = (set(header) | set(refined)) - set(ids)
        raise ValueError(f'Manifest mismatch: missing={len(missing)}, extra={len(extra)}')
    paired = []
    for c in cases:
        key = c['relative_dir']
        h, r = header[key], refined[key]
        if h.get('status') != 'ok' or r.get('status') != 'ok':
            raise ValueError(f'Incomplete case: {key}')
        t0, t1 = float(h['mean_tre_mm']), float(r['mean_tre_mm'])
        if not np.isfinite([t0, t1, float(r['rot_deg']), float(r['trans_mm'])]).all():
            raise ValueError(f'Nonfinite metrics: {key}')
        name = key.split('/')[-1]
        paired.append({
            'case': key, 'subject': c.get('subject', key.split('/')[1].split('_')[0]),
            'orientation': 'coronal' if name.endswith('_cor') else 'axial',
            'thickness_mm': int(name.split('_')[0][1:]),
            'position': name.split('_')[1],
            'start': 'large' if '_large' in name else 'small',
            'header_tre_mm': t0, 'refined_tre_mm': t1,
            'delta_tre_mm': t1 - t0, 'improved': t1 < t0,
            'rotation_error_deg': float(r['rot_deg']),
            'centroid_translation_error_mm': float(r['trans_mm']),
            'failure': bool(r['failure']), 'runtime_s': float(r['runtime_s']),
            'initial_objective': r.get('initial_objective'),
            'final_objective': r.get('final_objective'),
            'optimization_accepted': r.get('accepted'),
        })
    n = len(paired)
    method_tre = [r['refined_tre_mm'] for r in paired]
    summary = {
        'split': 'validation', 'cases': n, 'subjects': len({r['subject'] for r in paired}),
        'median_header_tre_mm': robust_median([r['header_tre_mm'] for r in paired]),
        'median_refined_tre_mm': robust_median(method_tre),
        'p90_refined_tre_mm': percentile(method_tre, 90),
        'p95_refined_tre_mm': percentile(method_tre, 95),
        'max_refined_tre_mm': max(method_tre),
        'failure_count': sum(r['failure'] for r in paired),
        'failure_pct': 100 * np.mean([r['failure'] for r in paired]),
        'tre_below_2mm_pct': 100 * np.mean([x < 2 for x in method_tre]),
        'paired_improved_count': sum(r['improved'] for r in paired),
        'paired_improved_pct': 100 * np.mean([r['improved'] for r in paired]),
        'median_rotation_error_deg': robust_median([r['rotation_error_deg'] for r in paired]),
        'median_centroid_translation_error_mm': robust_median([r['centroid_translation_error_mm'] for r in paired]),
        'median_runtime_s': robust_median([r['runtime_s'] for r in paired]),
        'tre_above_5_count': sum(x > 5 for x in method_tre),
        'tre_above_10_count': sum(x > 10 for x in method_tre),
        'tre_above_20_count': sum(x > 20 for x in method_tre),
        'failure_definition': 'rotation_error > 2 deg OR centroid_translation_error > 2 mm',
    }
    groups = []
    for key in ('orientation', 'start', 'thickness_mm', 'position', 'subject'):
        groups.extend(group_stats(paired, key))
    return summary, paired, groups


def save_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError('Cannot write empty table')
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--header', type=Path, required=True)
    p.add_argument('--refined', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    summary, paired, groups = audit(json.loads(a.manifest.read_text()),
                                    read_jsonl(a.header, 'initial_header'),
                                    read_jsonl(a.refined, 'refined_header'))
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    save_csv(a.output / 'paired_cases.csv', paired)
    save_csv(a.output / 'subgroups.csv', groups)
    save_csv(a.output / 'worst_cases.csv', sorted(paired, key=lambda x: -x['refined_tre_mm'])[:40])
    save_csv(a.output / 'regressions.csv', sorted((r for r in paired if r['delta_tre_mm'] > 0), key=lambda x: -x['delta_tre_mm']))
    print(json.dumps(summary, indent=2), flush=True)
    print(f'Wrote {len(paired)} paired cases; {len(groups)} subgroup rows', flush=True)


if __name__ == '__main__':
    main()
