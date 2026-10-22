"""Analyses on the 1,080 validation slabs, logged to W&B and saved to runs/deformable/analysis/.

mi-landscape       Does the slab-T1 mutual information peak at the true shift? MI between the slab and the T1 moved by
                   s x (true shift), s = 0, 0.25, ..., 2 (s = 0: no correction, s = 1: the truth). No training.
rigid-sensitivity  How exact must the rigid transform be? Trained models get the T1 resampled through the exact rigid
                   transform plus a random rigid error of 0, 1-2, 3-6 or 6-10 mm and degrees (the same error for
                   every model). Shift error = scored with the exact transform (the deformable stage alone);
                   total error = scored with the transform it was given.
examples           Slabs at the 10th, 50th and 90th percentile of a trained model's TRE (exact rigid transform):
                   T1 edges over the slab without and with the predicted shift; predicted and true shift along PE.
Run from scripts/:  python -m deformable.analysis mi-landscape
                    python -m deformable.analysis rigid-sensitivity ../runs/deformable/00_final ../runs/deformable/12_rigid_noise
                    python -m deformable.analysis examples ../runs/deformable/00_final
"""
import argparse
import csv
import json
import zlib
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader

from slabsim.geometry import random_rigid

from .data import DATA, ValSlabs, exact_M_val
from .evaluate import load_run, save
from .losses import mutual_information, voxel_errors
from .model import device, identity_grid, predict, warp
from .train import OUT

SCALES = np.linspace(0, 2, 9)
ERRORS = [(0, 0), (1, 2), (3, 6), (6, 10)]


def true_shift(b):
    """True shift (1, 3, X, Y, Z) in slab voxels: truth = M @ H @ (ijk + shift)."""
    A = torch.linalg.inv(b['M'] @ b['H']).float()
    ijk = (b['truth'] @ A[:3, :3].T + A[:3, 3]).permute(3, 0, 1, 2)
    return (ijk - identity_grid(ijk.shape[1:], 'cpu'))[None]


@torch.no_grad()
def mi_landscape(dev):
    rows = []
    for b in DataLoader(ValSlabs(), batch_size=None, num_workers=4):
        x, u = b['x'][None].to(dev), true_shift(b).to(dev)
        weight = (b['t1mask'].to(dev) & (x[0, 0] > 0.05)).float()[None, None]          # as in the MI loss
        mi = [mutual_information(x[:, :1], warp(x[:, 1:], s * u, padding='border'), weight,
                                 gen=torch.Generator().manual_seed(0)).item() for s in SCALES]  # same voxels for all s
        rows.append({'slab': b['name'], **b['meta'], 'best_scale': float(SCALES[int(np.argmax(mi))]),
                     **{f'mi_s{s:g}': m for s, m in zip(SCALES, mi)}})
    best = np.array([r['best_scale'] for r in rows])
    summary = {'pct_peak_at_truth': 100 * float(np.mean(best == 1)),
               'pct_peak_below_truth': 100 * float(np.mean(best < 1)),
               'pct_peak_above_truth': 100 * float(np.mean(best > 1)),
               'pct_truth_beats_no_correction': 100 * float(np.mean([r['mi_s1'] > r['mi_s0'] for r in rows]))}
    curve = [[s, float(np.median([r[f'mi_s{s:g}'] - r['mi_s1'] for r in rows]))] for s in SCALES]
    return rows, summary, curve


@torch.no_grad()
def rigid_sensitivity(runs, dev):
    rows = []
    for run in runs:
        net, cfg = load_run(run, dev)
        for lo, hi in ERRORS:
            shift_err, total_err, no_def = [], [], []
            for b in DataLoader(ValSlabs(rigid=perturbed(lo, hi)), batch_size=None, num_workers=4):
                u, _ = predict(net, cfg, b['xr'], b['pe_axis'], b['pe_sign'])
                shift_err.append(voxel_errors(u, b, b['M']).mean().item())
                total_err.append(voxel_errors(u, b, b['Mr']).mean().item())
                no_def.append(voxel_errors(torch.zeros_like(u), b, b['Mr']).mean().item())
            rows.append({'model': Path(run).name, 'rigid_error': f'{lo}-{hi}', 'shift_error': float(np.median(shift_err)),
                         'total_error': float(np.median(total_err)), 'no_deformation': float(np.median(no_def))})
            print(rows[-1], flush=True)
    return rows


@torch.no_grad()
def examples(run, dev, quantiles=(0.1, 0.5, 0.9)):
    """Figure: one row per slab at the given quantiles of the run's validation TRE, middle slice. Columns: T1 edges over
    the slab without / with the predicted shift (TRE of the whole slab in the titles); predicted / true shift along PE
    (mm) over the scored voxels."""
    net, cfg = load_run(run, dev)
    with open(Path(run) / 'val_slabs.csv') as f:
        rows = sorted(csv.DictReader(f), key=lambda r: float(r['tre']))
    ds = ValSlabs()
    names = [f'{d.parent.name}/{d.name}' for d in ds.dirs]
    fig, axes = plt.subplots(len(quantiles), 4, figsize=(13, 3.2 * len(quantiles)), layout='constrained')
    for ax, q in zip(axes, quantiles):
        r = rows[round(q * (len(rows) - 1))]
        b = ds[names.index(r['slab'])]
        u, _ = predict(net, cfg, b['x'], b['pe_axis'], b['pe_sign'])
        moved = warp(b['x'][None, 1:].to(dev), u)[0, 0].cpu()
        mm = b['H'][:3, b['pe_axis']].norm().item()                               # voxel size along PE
        along_pe = lambda s: np.where(b['mask'].numpy(), s[0, b['pe_axis']].cpu().numpy() * mm, np.nan)
        pred, true = along_pe(u), along_pe(true_shift(b))
        k, lim = b['x'].shape[3] // 2, np.nanpercentile(np.abs(true), 99)
        for a, t1, tre, title in [(ax[0], b['x'][1], r['no_deformation'], 'without deformation'),
                                  (ax[1], moved, r['tre'], 'with the predicted shift')]:
            a.imshow(b['x'][0, :, :, k].T, cmap='gray', origin='lower', vmin=0, vmax=1)
            a.contour(t1[:, :, k].T, levels=[0.7], colors='yellow', linewidths=0.8)       # grey / white matter boundary
            a.set_title(f'{title}: TRE {float(tre):.2f} mm', fontsize=10)
        for a, s, title in [(ax[2], pred, 'predicted shift along PE'), (ax[3], true, 'true shift along PE')]:
            a.set_facecolor('#e8e8e6')
            im = a.imshow(s[:, :, k].T, cmap='RdBu_r', origin='lower', vmin=-lim, vmax=lim)
            a.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax[2:], label='mm', shrink=0.9)
        ax[0].set_ylabel(f'{q:.0%} of slabs have lower TRE\n{r["orientation"]}, {r["thickness"]} mm, '
                         f'position {float(r["position"]):.0%}', fontsize=9)
        for a in ax:
            a.set_xticks([])
            a.set_yticks([])
    return fig


def perturbed(lo, hi):
    """{slab: exact rigid transform with a random rigid error of lo-hi mm and degrees}, seeded by the slab name."""
    out = {}
    for d in sorted((DATA / 'test').glob('*/t*')):
        p = json.loads((d / 'params.json').read_text())
        name = f'{d.parent.name}/{d.name}'
        rng = np.random.default_rng(zlib.crc32(name.encode()))
        out[name] = random_rigid(rng, lo, hi, p['center_t1_mm']) @ exact_M_val(p) if hi else exact_M_val(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('analysis', choices=['mi-landscape', 'rigid-sensitivity', 'examples'])
    ap.add_argument('runs', nargs='*', help='training run folders (rigid-sensitivity, examples)')
    args = ap.parse_args()
    dev = device()
    out = OUT / 'analysis'
    out.mkdir(parents=True, exist_ok=True)
    run = wandb.init(project='dlir-deformable', name=f'analysis_{args.analysis}', dir=out)
    if args.analysis == 'mi-landscape':
        rows, summary, curve = mi_landscape(dev)
        save(rows, summary, out, 'mi_landscape')
        run.summary.update(summary)
        table = wandb.Table(columns=['s', 'median MI(s) - MI(1)'], data=curve)
        run.log({'mi_landscape': wandb.plot.line(table, 's', 'median MI(s) - MI(1)'),
                 'mi_slabs': wandb.Table(columns=list(rows[0]), data=[list(r.values()) for r in rows])})
    elif args.analysis == 'examples':
        examples(args.runs[0], dev).savefig(out / 'examples.png', dpi=150)
        run.log({'examples': wandb.Image(str(out / 'examples.png'))})
    else:
        rows = rigid_sensitivity(args.runs, dev)
        save(rows, {}, out, 'rigid_sensitivity')
        run.log({'rigid_sensitivity': wandb.Table(columns=list(rows[0]), data=[list(r.values()) for r in rows])})
    run.finish()


if __name__ == '__main__':
    main()
