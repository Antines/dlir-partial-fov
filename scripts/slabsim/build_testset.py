"""Regime-A (semi-real) test slabs from one preprocessed run: real distorted SBRef slices with ground truth.

Grid: orientation x slab thickness x position (fraction of the brain's extent along the cut axis) x initial misalignment.
Axial slabs are cut along the SBRef's own slice axis; coronal slabs along its anterior-posterior axis (the volume is
re-indexed so that axis comes last; phase encoding stays in the slab plane). Coronal slab folders end in '_cor'.
Misalignment 'scanner' keeps the real header (for HCP: T1 and EPI from different sessions, often 10-25 mm off).
The other levels start from the best rigid fit of the ground truth (only the distortion remains) and add a random
rigid perturbation of 1-3 (small), 3-6 (moderate) or 6-10 (large) degrees and mm, applied to the slab header.
Usage: python build_testset.py <prep_run_dir> <anat_dir> <out_dir> [pilot|full]
"""
import json
import sys
import zlib
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slabsim.evaluation import fit_rigid
from slabsim.forward import real_slab_truth, sample
from slabsim.geometry import random_rigid, voxel_centers
from slabsim.samples import save_sample
from slabsim.sampling import make_real_sample

GRIDS = {'pilot': {'thickness_mm': [16, 32, 48], 'position': [0.3, 0.5, 0.7], 'misalignment': ['small', 'large']},
         'val': {'thickness_mm': [16, 32, 48], 'position': [0.3, 0.5, 0.7], 'misalignment': ['small', 'large'],
                 'orientation': ['axial', 'coronal']},
         'full': {'thickness_mm': [16, 24, 32, 48], 'position': [0.25, 0.45, 0.65, 0.85],
                  'misalignment': ['scanner', 'small', 'moderate', 'large'], 'orientation': ['axial', 'coronal']}}
MISALIGNMENT = {'scanner': None, 'small': (1, 3), 'moderate': (3, 6), 'large': (6, 10)}


def brain_slices(sbref, field, anchor, brain_mask, pe_axis, pe_sign, trt):
    """First and last slice (along the slice axis) that contain brain, via the ground truth of the whole volume."""
    data, affine = sbref
    _, truth, _, _ = real_slab_truth(field, affine, data.shape, 0, data.shape[2], pe_axis, pe_sign, trt, anchor)
    per_slice = (sample(*brain_mask, truth, order=0) > 0.5).sum(axis=(0, 1))
    inside = np.flatnonzero(per_slice > 0.05 * per_slice.max())
    return inside[0], inside[-1]


def slab_axes(sbref_affine, pe_axis, orientation):
    """Voxel-axis order of the cut volume (slabs are cut along its last axis) and the PE axis in that order.
    The permutation is cyclic and flips no axis, so the PE sign is unchanged."""
    cut = 2 if orientation == 'axial' else int(np.argmax(np.abs(sbref_affine[1, :3])))   # coronal: axis closest to world y
    perm = [(cut + 1) % 3, (cut + 2) % 3, cut]                    # cyclic: keeps the handedness of the voxel grid
    return perm, perm.index(pe_axis)


if __name__ == '__main__':
    run_dir, anat_dir, out_dir = map(Path, sys.argv[1:4])
    grid = GRIDS[sys.argv[4] if len(sys.argv) > 4 else 'pilot']
    acq = json.loads((run_dir / 'acq.json').read_text())
    raw, fld = nib.load(run_dir / 'sbref_raw.nii.gz'), nib.load(run_dir / 'field_hz_sbref.nii.gz')
    mask = nib.load(anat_dir / 'T1_mask.nii.gz')
    field, brain = (fld.get_fdata(), fld.affine), (mask.get_fdata(), mask.affine)
    anchor = np.loadtxt(run_dir / 'anchor_epi2t1.txt')
    seed = zlib.crc32(f'{run_dir.parent.name}/{run_dir.name}'.encode())   # fixed per subject and run, on any machine
    n_slabs = np.prod([len(grid[k]) for k in ('thickness_mm', 'position', 'misalignment')])
    for orientation in grid.get('orientation', ['axial']):
        suffix = '' if orientation == 'axial' else '_cor'
        built = [d for d in out_dir.glob('t*/slab.nii.gz') if d.parent.name.endswith('_cor') == bool(suffix)]
        if len(built) == n_slabs:
            continue                                              # resume: same seeds would rebuild identical slabs
        rng = np.random.default_rng(seed if orientation == 'axial' else [seed, 1])
        perm, pe_axis = slab_axes(raw.affine, acq['pe_axis'], orientation)
        sbref = (raw.get_fdata().transpose(perm), raw.affine[:, perm + [3]])
        slice_mm = float(np.linalg.norm(sbref[1][:3, 2]))
        k_lo, k_hi = brain_slices(sbref, field, anchor, brain, pe_axis, acq['pe_sign'], acq['trt'])
        for thickness in grid['thickness_mm']:
            n = max(2, int(round(thickness / slice_mm)))
            for position in grid['position']:
                k0 = int(np.clip(round(k_lo + position * (k_hi - k_lo) - n / 2), 0, sbref[0].shape[2] - n))
                s = make_real_sample(sbref, field, anchor, brain, k0, k0 + n, acq['pe_sign'], acq['trt'], pe_axis=pe_axis)
                ok = s['brain'] & s['valid']
                rigid_truth = fit_rigid(voxel_centers(s['affine'], s['image'].shape)[ok], s['truth'][ok])
                header = s['affine']
                for level in grid['misalignment']:
                    pose = np.eye(4) if MISALIGNMENT[level] is None else \
                        random_rigid(rng, *MISALIGNMENT[level], s['truth'][ok].mean(0)) @ rigid_truth
                    s['affine'] = pose @ header
                    init = np.linalg.norm(s['truth'] - voxel_centers(s['affine'], s['image'].shape), axis=-1)[ok]
                    params = {'orientation': orientation, 'thickness_mm': thickness, 'n_slices': n, 'position': position,
                              'k0': k0, 'center_t1_mm': s['truth'][ok].mean(0).round(1).tolist(), 'misalignment': level,
                              'header_pose': pose.tolist(), 'run': str(run_dir), 'initial_error_mm': float(init.mean()),
                              'pe_axis': pe_axis, 'pe_sign': acq['pe_sign'], 'trt': acq['trt']}   # PE in the slab's grid
                    name = f't{thickness:02d}_p{int(position * 100):02d}_{level}{suffix}'
                    save_sample(s, out_dir / name, params)
                    print(name, s['image'].shape, f'initial error {init.mean():.2f} mm')
