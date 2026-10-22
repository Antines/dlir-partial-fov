"""Inference and scoring utilities for the rigid model."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from slabsim.evaluation import displacement_error, linear_estimate, rigid_errors
from slabsim.geometry import voxel_centers

from .rigid_data import SavedSlabDataset
from .rigid_model_spatial_fusion_v1 import RigidRegistrationNet


def score_matrix(matrix, sample):
    image = sample['slab'].numpy()[0]
    affine = sample['slab_affine'].numpy()
    truth, mask = sample['truth'], sample['mask']
    est = linear_estimate(np.asarray(matrix), affine, image.shape)
    out = {
        **displacement_error(est, truth, mask),
        **rigid_errors(est, truth, voxel_centers(affine, image.shape), mask),
    }
    out['success'] = bool(out['mean'] < 2.0)
    out['failure'] = bool(out['rot_deg'] > 2.0 or out['trans_mm'] > 2.0)
    return out


@torch.no_grad()
def evaluate_model(model, dataset: SavedSlabDataset, device, output_dir=None):
    model.eval()
    rows = []
    out_root = Path(output_dir) if output_dir else None
    if out_root:
        out_root.mkdir(parents=True, exist_ok=True)
    for i in range(len(dataset)):
        s = dataset[i]
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred = model(s['slab'], s['slab_affine'], s['t1'], s['t1_affine'])
        if device.type == 'cuda':
            torch.cuda.synchronize()
        runtime = time.perf_counter() - t0
        matrix = pred.matrix.detach().cpu().numpy()
        r = score_matrix(matrix, s)
        r.update({'runtime_s': runtime, 'slab_dir': s['slab_dir']})
        rows.append(r)
        if out_root:
            d = Path(s['slab_dir'])
            rel = Path(d.parent.name) / d.name
            od = out_root / rel
            od.mkdir(parents=True, exist_ok=True)
            np.savetxt(od / 'rigid.txt', matrix, fmt='%.9g')
            (od / 'result.json').write_text(json.dumps(r, indent=2))

    tre = np.array([r['mean'] for r in rows], float)
    fail = np.array([r['failure'] for r in rows], bool)
    rot = np.array([r['rot_deg'] for r in rows], float)
    trans = np.array([r['trans_mm'] for r in rows], float)
    runt = np.array([r['runtime_s'] for r in rows], float)
    summary = {
        'n': len(rows),
        'median_tre_mm': float(np.median(tre)),
        'mean_tre_mm': float(np.mean(tre)),
        'tre_lt_2_pct': float(100 * np.mean(tre < 2)),
        'failure_pct': float(100 * np.mean(fail)),
        'median_rot_deg': float(np.median(rot)),
        'median_trans_mm': float(np.median(trans)),
        'median_runtime_s': float(np.median(runt)),
    }
    if out_root:
        (out_root / 'summary.json').write_text(json.dumps(summary, indent=2))
    return summary, rows


def load_checkpoint(path, device='cpu'):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    cfg = ckpt['config']
    mcfg = cfg['model']
    model = RigidRegistrationNet(
        grid_shape_xyz=mcfg['grid_shape_xyz'],
        base_channels=mcfg['base_channels'],
        max_rotation_deg=mcfg['max_rotation_deg'],
        max_translation_mm=mcfg['max_translation_mm'],
    ).to(device)
    model.load_state_dict(ckpt['model'])
    return model, cfg, ckpt
