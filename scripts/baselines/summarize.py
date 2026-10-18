"""Summarize baseline results over test slabs and apply the pre-specified selection rule.

Rule: among classical methods, lowest median TRE over slabs, restricted to methods whose failure rate
(rigid part off by > 2 mm or > 2 deg) is <= 10 %; ties broken by runtime.
If no method meets the criterion, the lowest failure rate wins (added 2026-10-14 after the pilot, when no method did).
Usage: python summarize.py <testset_dir> [<testset_dir> ...]     (each containing slab folders with results.json)
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings('ignore', category=RuntimeWarning)       # empty groups (e.g. a method that never ran) -> nan

CLASSICAL = ['fsl_bbr', 'epi_reg', 'ants_rigid_mi', 'bridged_bbr', 'afni_lpc']

if __name__ == '__main__':
    rows = []
    for d in map(Path, sys.argv[1:]):
        for f in sorted(d.glob('*/results.json')):
            params = json.loads((f.parent / 'params.json').read_text())
            for method, r in json.loads(f.read_text()).items():
                rows.append({'set': d.name, 'slab': f.parent.name, 'method': method, **params, **r})
    methods = sorted({r['method'] for r in rows})
    for r in rows:
        r.setdefault('orientation', 'axial')                      # slabs built before coronal slabs existed
    groups = [('all', lambda r: True)] + \
             [(f'{o}', lambda r, o=o: r['orientation'] == o) for o in sorted({r['orientation'] for r in rows})] + \
             [(f'misalign {m}', lambda r, m=m: r['misalignment'] == m) for m in sorted({r['misalignment'] for r in rows})] + \
             [(f'thickness {t}', lambda r, t=t: r['thickness_mm'] == t) for t in sorted({r['thickness_mm'] for r in rows})] + \
             [(f'position {p}', lambda r, p=p: r['position'] == p) for p in sorted({r['position'] for r in rows})]
    print(f'{len({(r["set"], r["slab"]) for r in rows})} slabs from {len(sys.argv) - 1} runs; cells: median TRE mm [failure %]')
    print(f"{'subset':16s}" + ''.join(f'{m[:15]:>18s}' for m in methods))
    for label, keep in groups:
        cells = []
        for m in methods:
            sel = [r for r in rows if r['method'] == m and keep(r)]
            ok = [r for r in sel if 'mean' in r]
            fail = [('mean' not in r) or r['rot_deg'] > 2 or r['trans_mm'] > 2 for r in sel]
            cells.append(f"{np.median([r['mean'] for r in ok]) if ok else np.nan:8.2f} [{100 * np.mean(fail):3.0f}%]  ")
        print(f'{label:16s}' + ''.join(f'{c:>18s}' for c in cells))
    runtime = {m: np.median([r['runtime_s'] for r in rows if r['method'] == m and 'runtime_s' in r]) for m in methods}
    print('median runtime s:', {m: round(float(v), 1) for m, v in runtime.items()})
    candidates = []
    for m in CLASSICAL:
        sel = [r for r in rows if r['method'] == m]
        if sel:
            fail = np.mean([('mean' not in r) or r['rot_deg'] > 2 or r['trans_mm'] > 2 for r in sel])
            tre = np.median([r['mean'] for r in sel if 'mean' in r])
            candidates.append((fail, tre, runtime[m], m))
    ok = [c for c in candidates if c[0] <= 0.10]
    best = min(ok, key=lambda c: (c[1], c[2])) if ok else min(candidates)
    print(f'selection rule -> {best[3]} (median TRE {best[1]:.2f} mm, failure {100 * best[0]:.0f} %)'
          + ('' if ok else '  NOTE: no method meets the <= 10 % failure criterion; lowest failure rate chosen'))
