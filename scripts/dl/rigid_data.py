"""Training and validation data adapters for the rigid DL stage."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from slabsim.dataset import Run
from slabsim.evaluation import fit_rigid
from slabsim.geometry import apply, random_rigid, voxel_centers
from slabsim.samples import load_sample
from slabsim.sampling import draw_params, make_sample


def _fit_target(affine, shape, truth, mask, rng, max_fit_points=25000):
    points = voxel_centers(affine, shape)
    p, q = points[mask], truth[mask]
    if len(p) < 16:
        raise RuntimeError("too few valid brain voxels for a rigid target")
    if len(p) > max_fit_points:
        idx = rng.choice(len(p), max_fit_points, replace=False)
        p, q = p[idx], q[idx]
    return fit_rigid(p, q)


def _probe_points(affine, shape, mask, target, rng, n=2048):
    points = voxel_centers(affine, shape)[mask]
    if len(points) == 0:
        raise RuntimeError("empty registration mask")
    idx = rng.choice(len(points), n, replace=len(points) < n)
    src = points[idx].astype(np.float32)
    dst = apply(target, src).astype(np.float32)
    return src, dst


def _easy_params(run: Run, rng: np.random.Generator):
    """Curriculum start used for controlled training: thick slabs, small tilt/start error."""
    p = draw_params(rng, run.center, run.extent)
    cor = p['orientation'] == 'coronal'
    p['thickness'] = float(rng.uniform(48, 60))
    p['angles'] = (
        float((90 if cor else 0) + rng.uniform(-5, 5)),
        float(rng.uniform(-5, 5)),
        float(rng.uniform(-5, 5)),
    )
    p['header_pose'] = random_rigid(rng, 0, 4, p['center']).tolist()
    return p


class SlabStream(IterableDataset):
    """Infinite on-the-fly simulated slabs with rigid targets from dense truth."""

    def __init__(self, runs: Iterable[Path], seed=0, slabs_per_run=8, difficulty='easy', probe_points=2048):
        self.runs = [Path(r) for r in runs]
        self.seed = int(seed)
        self.slabs_per_run = int(slabs_per_run)
        self.difficulty = str(difficulty)
        self.probe_points = int(probe_points)
        if self.difficulty not in {'easy', 'full'}:
            raise ValueError("difficulty must be 'easy' or 'full'")

    def __iter__(self):
        info = get_worker_info()
        worker = info.id if info else 0
        rng = np.random.default_rng([self.seed, worker])
        while True:
            run_dir = self.runs[int(rng.integers(len(self.runs)))]
            run = Run(run_dir)
            for _ in range(self.slabs_per_run):
                if self.difficulty == 'easy':
                    params = _easy_params(run, rng)
                    s = {**make_sample(run.epi, run.field, run.mask, params, rng), 'params': params}
                else:
                    s = run.sample(rng)
                    params = s['params']
                mask = s['brain'] & s['valid']
                if mask.sum() < 64:
                    continue
                target = _fit_target(s['affine'], s['image'].shape, s['truth'], mask, rng)
                src_probe, dst_probe = _probe_points(
                    s['affine'], s['image'].shape, mask, target, rng, self.probe_points
                )
                yield {
                    'slab': torch.from_numpy(s['image'][None].astype(np.float32)),
                    'slab_affine': torch.from_numpy(s['affine'].astype(np.float32)),
                    't1': torch.from_numpy(run.t1[0][None].astype(np.float32)),
                    't1_affine': torch.from_numpy(run.t1[1].astype(np.float32)),
                    'target_matrix': torch.from_numpy(target.astype(np.float32)),
                    'probe_points': torch.from_numpy(src_probe),
                    'target_probe_points': torch.from_numpy(dst_probe),
                    'params': {
                        'orientation': params['orientation'],
                        'thickness': float(params['thickness']),
                        'pe_axis': int(params['pe_axis']),
                        'pe_sign': int(params['pe_sign']),
                        'trt': float(params['trt']),
                    },
                    'run_dir': str(run_dir),
                }


@lru_cache(maxsize=16)
def _load_t1_cached(path: str):
    img = nib.load(path)
    return img.get_fdata(dtype=np.float32), img.affine.astype(np.float32)


def _subject_from_group(group_name: str) -> str:
    marker = '_rfMRI_'
    if marker not in group_name:
        raise ValueError(f"cannot infer subject from validation group {group_name}")
    return group_name.split(marker, 1)[0]


class SavedSlabDataset(Dataset):
    """Saved real-distortion validation slabs from ``dlir_data/test``."""

    def __init__(self, data_root, slab_dirs=None, max_slabs=None):
        self.data_root = Path(data_root)
        if slab_dirs is None:
            slab_dirs = sorted(d for d in (self.data_root / 'test').glob('*/t*') if d.is_dir())
        self.slab_dirs = [Path(d) for d in slab_dirs]
        if max_slabs:
            # deterministic coverage of the entire sorted validation set
            idx = np.linspace(0, len(self.slab_dirs) - 1, min(int(max_slabs), len(self.slab_dirs))).round().astype(int)
            self.slab_dirs = [self.slab_dirs[i] for i in idx]

    def __len__(self):
        return len(self.slab_dirs)

    def __getitem__(self, idx):
        d = self.slab_dirs[idx]
        image, affine, truth, mask = load_sample(d)
        params = json.loads((d / 'params.json').read_text())
        subject = _subject_from_group(d.parent.name)
        t1, t1_affine = _load_t1_cached(str(self.data_root / 'data' / subject / 'anat' / 'T1.nii.gz'))
        return {
            'slab': torch.from_numpy(image[None].astype(np.float32)),
            'slab_affine': torch.from_numpy(affine.astype(np.float32)),
            't1': torch.from_numpy(t1[None].astype(np.float32)),
            't1_affine': torch.from_numpy(t1_affine),
            'truth': truth,
            'mask': mask,
            'params': params,
            'slab_dir': str(d),
        }
