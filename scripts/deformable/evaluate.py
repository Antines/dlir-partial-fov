"""Score a trained deformable stage on real-distortion slabs.

TRE of a slab = mean distance (mm) between the predicted and the true T1 position over its scored voxels; the summary
gives medians over slabs. Two rigid inputs: the exact rigid transform (isolates the deformable stage) and, with
--rigid, the rigid model's transform of every slab (the realistic cascade).
Run from scripts/:  python -m deformable.evaluate ../runs/deformable/00_final [--slabs DIR] [--rigid CSV]
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .data import DATA, ValSlabs, read_rigid
from .losses import voxel_errors
from .model import build, jacobian_det, predict


def load_run(run_dir, device):
    """Network with the chosen checkpoint (best.pt) of a training run, and its config."""
    cfg = yaml.safe_load((Path(run_dir) / 'config.yaml').read_text())
    net = build(cfg).to(device)
    net.load_state_dict(torch.load(Path(run_dir) / 'best.pt', map_location=device))
    return net.eval(), cfg


@torch.no_grad()
def evaluate(net, cfg, slabs=DATA / 'test', rigid=None, workers=4):
    """Per-slab errors (list of dicts) and their summary."""
    net.eval()
    rows = []
    for b in DataLoader(ValSlabs(slabs, rigid), batch_size=None, num_workers=workers):
        u, _ = predict(net, cfg, b['x'], b['pe_axis'], b['pe_sign'])
        e = voxel_errors(u, b, b['M'])
        row = {'slab': b['name'], **b['meta'], 'tre': e.mean().item(),
               'tre_p95': torch.quantile(e.float().cpu(), 0.95).item(),
               'no_deformation': voxel_errors(torch.zeros_like(u), b, b['M']).mean().item(),
               'fold_pct': 100 * (jacobian_det(u)[b['mask'].to(u.device)] <= 0).float().mean().item()}
        if rigid is not None:
            ur, _ = predict(net, cfg, b['xr'], b['pe_axis'], b['pe_sign'])
            row['tre_rigid_model'] = voxel_errors(ur, b, b['Mr']).mean().item()
            row['no_deformation_rigid_model'] = voxel_errors(torch.zeros_like(ur), b, b['Mr']).mean().item()
        rows.append(row)
    net.train()
    return rows, summarize(rows)


def summarize(rows):
    """Medians over slabs: overall, per orientation / position / thickness; share of slabs under 2 mm; mean share of
    folded voxels; the reference methods stored with the slabs (base_*)."""
    med = lambda k, sel=lambda r: True: float(np.median([r[k] for r in rows if sel(r)]))
    s = {'tre': med('tre'), 'tre_p95': med('tre_p95'), 'no_deformation': med('no_deformation'),
         'pct_below_2mm': 100 * float(np.mean([r['tre'] < 2 for r in rows])),
         'fold_pct': float(np.mean([r['fold_pct'] for r in rows]))}
    for key in ('orientation', 'position', 'thickness'):
        for val in sorted({r[key] for r in rows}):
            s[f'tre_{key}_{val}'] = med('tre', lambda r: r[key] == val)
    for k in rows[0]:
        if k.startswith('base_') or k.endswith('rigid_model'):
            s[k] = med(k)
    if 'tre_rigid_model' in rows[0]:
        s['pct_below_2mm_rigid_model'] = 100 * float(np.mean([r['tre_rigid_model'] < 2 for r in rows]))
    return s


def save(rows, summary, out, tag):
    with open(Path(out) / f'{tag}_slabs.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (Path(out) / f'{tag}_summary.json').write_text(json.dumps(summary, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run', help='training run folder with config.yaml and best.pt')
    ap.add_argument('--slabs', default=str(DATA / 'test'), help='folder of real-distortion slabs')
    ap.add_argument('--rigid', help="CSV with the rigid model's transform of every slab (see data.read_rigid)")
    args = ap.parse_args()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net, cfg = load_run(args.run, dev)
    rows, summary = evaluate(net, cfg, args.slabs, read_rigid(args.rigid) if args.rigid else None)
    save(rows, summary, args.run, Path(args.slabs).name)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
