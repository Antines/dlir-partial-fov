#!/usr/bin/env python3
"""Export qualitative EPI/T1 overlays for the worst completed validation cases."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates


def resample_t1(t1: np.ndarray, t1_affine: np.ndarray, slab_affine: np.ndarray,
                slab_shape: tuple[int, int, int]) -> np.ndarray:
    ijk = np.indices(slab_shape, dtype=np.float32).reshape(3, -1)
    world = slab_affine[:3, :3] @ ijk + slab_affine[:3, 3:4]
    vox = np.linalg.solve(t1_affine[:3, :3], world - t1_affine[:3, 3:4])
    values = map_coordinates(t1, vox, order=1, mode='constant', cval=0.0)
    return values.reshape(slab_shape)


def robust_contrast(img: np.ndarray):
    good = np.isfinite(img) & (img > 0)
    if np.count_nonzero(good) < 16:
        return np.zeros_like(img)
    a, b = np.percentile(img[good], [2, 98])
    return np.clip((img - a) / max(b - a, 1e-6), 0, 1)


def plot_case(data: Path, case_id: str, transform_meta: dict, output_png: Path):
    import nibabel as nib
    d = data / case_id
    epi_img = nib.load(str(d / 'slab.nii.gz'))
    epi = epi_img.get_fdata(dtype=np.float32)
    subject = case_id.split('/')[1].split('_')[0]
    fixed_img = nib.load(str(data / 'data' / subject / 'anat' / 'T1.nii.gz'))
    fixed = fixed_img.get_fdata(dtype=np.float32)
    original = resample_t1(fixed, fixed_img.affine, epi_img.affine, epi.shape)
    aligned = resample_t1(fixed, fixed_img.affine,
                          np.array(transform_meta['corrected_affine']), epi.shape)
    z = epi.shape[2] // 2
    panels = [(epi[:, :, z], 'EPI slab'), (original[:, :, z], 'T1 from header'),
              (aligned[:, :, z], 'T1 after rigid refinement')]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, (p, title) in zip(axes, panels):
        ax.imshow(robust_contrast(p).T, cmap='gray', origin='lower', vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis('off')
    fig.suptitle(f'{case_id} | central slab slice', fontsize=10)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, type=Path)
    p.add_argument('--worst-cases', required=True, type=Path)
    p.add_argument('--handoff', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--top', default=8, type=int)
    a = p.parse_args()
    with a.worst_cases.open() as f:
        rows = list(csv.DictReader(f))
    for i, r in enumerate(rows[:a.top], 1):
        case = r['case']
        _, group, slab = case.split('/')
        meta_path = a.handoff / 'cases' / group / slab / 'metadata.json'
        if not meta_path.is_file():
            raise FileNotFoundError(meta_path)
        plot_case(a.data, case, json.loads(meta_path.read_text()),
                  a.output / f'{i:02d}_{group}_{slab}.png')
        print(f'[{i}/{min(a.top, len(rows))}] QC: {case}', flush=True)


if __name__ == '__main__':
    main()
