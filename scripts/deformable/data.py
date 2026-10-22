"""Data of the deformable stage, on top of the slab simulator (slabsim).

Every sample lives on the slab's own voxel grid:
  x        (2, X, Y, Z)  slab and T1; the T1 is resampled at M @ H @ ijk (H = slab header, M = rigid transform)
  truth    (X, Y, Z, 3)  true T1 position (mm) of every slab voxel
  mask     (X, Y, Z)     scored voxels: brain and unambiguous distortion
  t1mask   (X, Y, Z)     T1 brain mask on the slab grid (weights the MI loss; needs no ground truth)
With the exact M the truth is M @ H @ (ijk + shift along the PE axis), so the network only has to find that shift.
"""
import csv
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from slabsim.dataset import Run, usable_runs
from slabsim.forward import sample
from slabsim.geometry import apply, random_rigid, rigid, voxel_centers
from slabsim.samples import load_sample

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get('DLIR_DATA', REPO / 'dlir_data'))


def split_runs(n_holdout=10, seed=0):
    """Training runs, and the runs of 10 training subjects held out for choosing the checkpoint."""
    ids = (REPO / 'splits' / 'hcp_train.txt').read_text().split()
    hold = set(np.random.default_rng(seed).choice(ids, n_holdout, replace=False).tolist())
    return (usable_runs(DATA / 'data', [i for i in ids if i not in hold]),
            usable_runs(DATA / 'data', sorted(hold)))


def exact_M_train(params):
    """Rigid part of the truth of a simulated slab: head motion and header start error undone."""
    pose = rigid(params['head_rot_deg'], params['head_trans_mm'], center=params['center'])
    return np.linalg.inv(pose) @ np.linalg.inv(np.asarray(params['header_pose']))


def exact_M_val(params):
    """Rigid part of the truth of a real-distortion slab: whole-brain anchor and header start error undone."""
    return np.loadtxt(DATA / params['run'] / 'anchor_epi2t1.txt') @ np.linalg.inv(np.asarray(params['header_pose']))


def t1_scale(t1):
    return float(np.percentile(t1[0][t1[0] > 0], 99))


def make_inputs(image, H, M, t1, t1mask, scale):
    """Slab and T1 on the slab grid, each divided by its 99th percentile; T1 brain mask on the slab grid."""
    pts = apply(M @ H, voxel_centers(np.eye(4), image.shape))
    pos = image[image > 0]
    slab = image / (np.percentile(pos, 99) if pos.size else 1.0)
    x = np.stack([slab, sample(*t1, pts, order=1) / scale]).astype(np.float32)
    return x, sample(*t1mask, pts, order=0) > 0.5


def to_torch(d):
    return {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in d.items()}


def simulated(run, scale, rng, rigid_noise=0.0):
    """One random training slab of a run (None if fewer than 500 voxels are scored); scale = t1_scale(run.t1).
    rigid_noise > 0 resamples the T1 through M with an extra random rigid error of up to rigid_noise mm and degrees."""
    s = run.sample(rng)
    p, mask = s['params'], s['brain'] & s['valid']
    if mask.sum() < 500:
        return None
    M = exact_M_train(p)
    if rigid_noise > 0:
        M = random_rigid(rng, 0.0, rigid_noise, s['truth'][mask].mean(0)) @ M
    x, t1m = make_inputs(s['image'], s['affine'], M, run.t1, run.mask, scale)
    return to_torch({'x': x, 'truth': s['truth'], 'mask': mask, 't1mask': t1m, 'H': s['affine'], 'M': M,
                     'pe_axis': int(p['pe_axis']), 'pe_sign': int(p['pe_sign'])})


class TrainStream(IterableDataset):
    """Endless random training slabs: each DataLoader worker loads a random run and simulates 6 slabs from it."""

    def __init__(self, runs, seed=0, rigid_noise=0.0):
        self.runs, self.seed, self.rigid_noise = runs, seed, rigid_noise

    def __iter__(self):
        info = get_worker_info()
        rng = np.random.default_rng([self.seed, info.id if info else 0])
        while True:
            run = Run(self.runs[rng.integers(len(self.runs))])
            scale = t1_scale(run.t1)
            for _ in range(6):
                s = simulated(run, scale, rng, self.rigid_noise)
                if s is not None:
                    yield s


def holdout_set(runs, cache, n=64, seed=123):
    """64 fixed simulated slabs from the held-out runs (exact M), cached on disk."""
    if Path(cache).exists():
        return torch.load(cache, weights_only=False)
    rng, out = np.random.default_rng(seed), []
    while len(out) < n:
        run = Run(runs[len(out) % len(runs)])
        s = simulated(run, t1_scale(run.t1), rng)
        if s is not None:
            out.append(s)
    Path(cache).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, cache)
    return out


def read_rigid(path):
    """CSV with columns slab, m00 ... m23 (the top 3 rows of a 4 x 4 matrix) -> {slab: M}."""
    with open(path) as f:
        return {r['slab']: np.vstack([np.array([float(r[f'm{i}{j}']) for i in range(3) for j in range(4)]).reshape(3, 4),
                                      [0, 0, 0, 1]]) for r in csv.DictReader(f)}


class ValSlabs(torch.utils.data.Dataset):
    """Real-distortion slabs of a folder (default: the 1,080 validation slabs). The T1 is resampled through the exact
    rigid transform (x, M) and, if `rigid` maps slab names to 4 x 4 matrices, also through those (xr, Mr)."""

    def __init__(self, root=DATA / 'test', rigid=None):
        self.dirs = sorted(d for d in Path(root).glob('*/t*') if d.is_dir())
        self.rigid = rigid
        self.anat = {}

    def __len__(self):
        return len(self.dirs)

    def load_anat(self, subject):
        if subject not in self.anat:                    # one subject at a time: slabs come sorted by subject
            load = lambda f: (nib.load(f).get_fdata(dtype=np.float32), nib.load(f).affine)
            t1 = load(DATA / 'data' / subject / 'anat' / 'T1.nii.gz')
            self.anat = {subject: (t1, load(DATA / 'data' / subject / 'anat' / 'T1_mask.nii.gz'), t1_scale(t1))}
        return self.anat[subject]

    def __getitem__(self, i):
        d = self.dirs[i]
        name = f'{d.parent.name}/{d.name}'
        image, H, truth, mask = load_sample(d)
        p = json.loads((d / 'params.json').read_text())
        res = json.loads((d / 'results.json').read_text())
        t1, t1mask, scale = self.load_anat(Path(p['run']).parts[1])
        M = exact_M_val(p)
        x, t1m = make_inputs(image, H, M, t1, t1mask, scale)
        meta = {'orientation': p['orientation'], 'thickness': p['thickness_mm'], 'position': p['position'],
                'start_error': p['misalignment'],
                **{f'base_{k}': res[k]['mean'] for k in ('header', 'rigid_oracle', 'affine_oracle', 'afni_lpc')
                   if 'mean' in res.get(k, {})}}
        item = {'name': name, 'x': x, 'truth': truth, 'mask': mask, 't1mask': t1m, 'H': H, 'M': M,
                'pe_axis': int(p['pe_axis']), 'pe_sign': int(p['pe_sign']), 'meta': meta}
        if self.rigid is not None:
            item['Mr'] = self.rigid[name]
            item['xr'] = make_inputs(image, H, item['Mr'], t1, t1mask, scale)[0]
        return to_torch(item)
