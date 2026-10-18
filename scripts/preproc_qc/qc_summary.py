"""Preprocessing QC table and summary: every run's acq.json -> qc_runs.csv (one row per run) and qc_summary.txt.
A run is included if it passes slabsim.dataset.passes_qc: anchor QC passed (BBR restarts within 0.5 mm),
gradient-nonlinearity residual <= 1.0 mm, field map reproduces its spin echoes (forward-check correlation >= 0.9).
If <out_dir>/forward_check.csv (validate_forward.py) exists, its summary is added.
Usage: python qc_summary.py <prep_hcp_dir> <splits_dir> <out_dir>
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slabsim.dataset import passes_qc

if __name__ == '__main__':
    prep, splits, out = map(Path, sys.argv[1:4])
    split = {s: n for n in ('train', 'val', 'test') for s in (splits / f'hcp_{n}.txt').read_text().split()}
    rows = []
    for acq in sorted(prep.glob('*/rfMRI_*/acq.json')):
        q = json.loads(acq.read_text())
        corr = {int(k): v for k, v in q['corr_by_sign'].items()}
        rows.append({'subject': acq.parent.parent.name, 'run': acq.parent.name, 'split': split.get(acq.parent.parent.name, ''),
                     'pe_dir': q['pe_dir'], 'pe_sign': q['pe_sign'], 'pe_sign_data_agrees': max(corr, key=corr.get) == q['pe_sign'],
                     'anchor_restart_max_mm': round(max(q['anchor_restart_mm']), 3), 'anchor_ok': q['anchor_ok'],
                     'gnl_residual_p95_mm': round(q['gnl_residual_p95_mm'], 3), 'field_hz_p99_abs': round(q['field_hz_p99_abs'], 1),
                     'header_to_anchor_mm': round(q['header_to_anchor_mm'], 2), 'fmap': q.get('fmap', ''),
                     'field_check_corr': q.get('field_check_corr', ''), 'included': passes_qc(q)})
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'qc_runs.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)

    col = lambda k: np.array([r[k] for r in rows], float)
    stats = lambda x: f'median {np.median(x):.2f}, 95th pct {np.percentile(x, 95):.2f}, max {x.max():.2f}'
    lines = [f'subjects {len({r["subject"] for r in rows})}, runs {len(rows)}',
             f'anchor QC passed: {sum(r["anchor_ok"] for r in rows)} / {len(rows)}; BBR restart spread (mm): {stats(col("anchor_restart_max_mm"))}',
             f'gradient-nonlinearity residual, 95th pct per run (mm): {stats(col("gnl_residual_p95_mm"))}; '
             f'> 1.0 mm (excluded): {int((col("gnl_residual_p95_mm") > 1.0).sum())}',
             f'PE sign: data check agrees with the HCP convention in {sum(r["pe_sign_data_agrees"] for r in rows)} / {len(rows)} runs',
             f'field map, strongest 1 % per run (Hz): {stats(col("field_hz_p99_abs"))}',
             f'scanner header vs anchor (mm): {stats(col("header_to_anchor_mm"))}',
             f'field-map check < 0.9 (excluded): {sum(r["field_check_corr"] != "" and r["field_check_corr"] < 0.9 for r in rows)} runs', '',
             'split   subjects  runs  included runs  subjects with an included run']
    for n in ('train', 'val', 'test'):
        sel = [r for r in rows if r['split'] == n]
        inc = [r for r in sel if r['included']]
        lines.append(f'{n:7s} {len({r["subject"] for r in sel}):8d} {len(sel):5d} {len(inc):14d} {len({r["subject"] for r in inc}):10d}')
    per = Counter(r['subject'] for r in rows if r['included'] and r['split'] in ('train', 'val'))
    lines.append('included runs per train/val subject: ' + ', '.join(f'{k} runs: {v} subjects' for k, v in sorted(Counter(per.values()).items(), reverse=True)))
    fc = out / 'forward_check.csv'
    if fc.exists():
        f = list(csv.DictReader(open(fc)))
        for sign, name in (('-1', 'LR'), ('1', 'RL')):
            sel = [r for r in f if r['pe_sign'] == sign]
            u, s = (np.array([float(r[k]) for r in sel]) for k in ('corr_undistorted', 'corr_simulated'))
            still = np.array([float(r['motion_mm']) <= 1 and float(r['motion_deg']) <= 1 for r in sel])
            lines.append(f'forward model, {name} spin echoes ({len(u)} field maps): correlation with the real image, median '
                         f'{np.median(u):.3f} undistorted -> {np.median(s):.3f} simulated; lowest {s.min():.3f} '
                         f'(head motion <= 1 mm and 1 deg: lowest {s[still].min():.3f}, {(~still).sum()} field maps moved more)')
    (out / 'qc_summary.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
