"""Group gradient-nonlinearity (GNL) field of the HCP scanner and its effect on each run's ground truth.

1. From the subjects with gnl_check.py output, map every displacement (raw -> corrected T1) to scanner coordinates
   and average across subjects on a 4 mm scanner grid (the field is scanner-fixed: cross-subject r ~ 0.8).
2. For every preprocessed run: the same anatomy sits at different scanner positions in the T1 and EPI sessions,
   so the raw images are distorted differently. The anchor absorbs the rigid part; the non-rigid rest is a
   ground-truth error. Its 95th percentile over brain voxels is stored as 'gnl_residual_p95_mm' in acq.json.
Usage: python gnl_group.py <prep_hcp_root>
Without gnl_check.py outputs under the root, an existing <prep_hcp_root>/gnl_group_field.nii.gz is reused (e.g. copied
from another machine; the field belongs to the scanner, not the subject).
"""
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slabsim.evaluation import fit_rigid
from slabsim.geometry import apply, voxel_centers
from slabsim.tools import ants_world

GRID = np.diag([4.0, 4.0, 4.0, 1.0])
GRID[:3, 3] = -120
SHAPE = (61, 61, 61)                                           # scanner coordinates -120 .. +120 mm


def subject_field(gnl_dir):
    """Displacement (raw -> corrected, mm, scanner axes) of one subject sampled on the scanner grid (NaN outside)."""
    fixed = nib.load(gnl_dir / 'dc_1mm.nii.gz')
    f = fixed.get_fdata()
    u = nib.load(gnl_dir / 'gnl_1Warp.nii.gz').get_fdata().squeeze() * np.array([-1, -1, 1])   # LPS -> RAS
    dc2raw = np.linalg.inv(ants_world(gnl_dir / 'gnl_0GenericAffine.mat'))
    head = f > np.percentile(f[f > 0], 25)
    p = apply(dc2raw, voxel_centers(fixed.affine, f.shape)[head])
    vec = u[head] @ dc2raw[:3, :3].T
    dist, idx = cKDTree(p[::5]).query(voxel_centers(GRID, SHAPE).reshape(-1, 3), distance_upper_bound=3)
    out = np.full((np.prod(SHAPE), 3), np.nan)
    out[np.isfinite(dist)] = vec[::5][idx[np.isfinite(dist)]]
    return out.reshape(*SHAPE, 3)


if __name__ == '__main__':
    root = Path(sys.argv[1])
    dirs = [d for d in sorted(root.glob('*/gnl')) if (d / 'gnl.json').exists()]
    if dirs:
        fields = np.stack([subject_field(d) for d in dirs])
        count = np.isfinite(fields[..., 0]).sum(0)
        mean = np.where(count[..., None] > 0, np.nansum(fields, 0) / np.maximum(count, 1)[..., None], 0.0)
        nib.save(nib.Nifti1Image(mean.astype(np.float32), GRID), root / 'gnl_group_field.nii.gz')
    else:
        mean = nib.load(root / 'gnl_group_field.nii.gz').get_fdata()
    D = RegularGridInterpolator([GRID[0, 3] + 4.0 * np.arange(n) for n in SHAPE], mean, bounds_error=False,
                                fill_value=0.0)
    for acq_file in sorted(root.glob('*/rfMRI_*/acq.json')):
        run = acq_file.parent
        epi = nib.load(run / 'sbref_corr.nii.gz')
        e = epi.get_fdata()
        p_epi = voxel_centers(epi.affine, e.shape)[e > np.percentile(e, 70)]
        p_t1 = apply(np.loadtxt(run / 'anchor_epi2t1.txt'), p_epi)
        moved = p_epi + D(p_epi) - D(p_t1)
        residual = np.linalg.norm(moved - apply(fit_rigid(p_epi, moved), p_epi), axis=1)
        qc = json.loads(acq_file.read_text())
        qc['gnl_residual_p95_mm'] = float(np.percentile(residual, 95))
        acq_file.write_text(json.dumps(qc, indent=1))
        print(run.parent.name, run.name, round(qc['gnl_residual_p95_mm'], 2), flush=True)
