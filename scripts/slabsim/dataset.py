"""Training data for the DL model: the usable runs of a split, and random regime-B slabs generated on the fly."""
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from .geometry import voxel_centers
from .sampling import draw_params, make_sample


def passes_qc(acq):
    """Inclusion rule for a run (its acq.json as a dict): anchor QC passed, gradient-nonlinearity residual <= 1 mm, and
    its field map reproduces its own spin echoes (forward-check correlation >= 0.9). A value not computed counts as
    passed."""
    return acq['anchor_ok'] and acq.get('gnl_residual_p95_mm', 0) <= 1.0 and acq.get('field_check_corr', 1) >= 0.9


def usable_runs(prep_root, subjects):
    """Run folders of these subjects that pass the QC."""
    return [acq.parent for s in subjects for acq in sorted(Path(prep_root, s).glob('rfMRI_*/acq.json'))
            if passes_qc(json.loads(acq.read_text()))]


def _load(f, to_t1=np.eye(4)):
    img = nib.load(f)
    return img.get_fdata(dtype=np.float32), to_t1 @ img.affine


class Run:
    """One run in T1 space: corrected EPI and field map (the anchor multiplied into their headers, no resampling),
    the subject's T1 (the fixed image) and brain mask."""

    def __init__(self, run_dir):
        run_dir = Path(run_dir)
        anchor = np.loadtxt(run_dir / 'anchor_epi2t1.txt')
        self.epi = _load(run_dir / 'sbref_corr.nii.gz', anchor)
        self.field = _load(run_dir / 'field_hz_sbref.nii.gz', anchor)
        self.t1 = _load(run_dir.parent / 'anat' / 'T1.nii.gz')
        self.mask = _load(run_dir.parent / 'anat' / 'T1_mask.nii.gz')
        pts = voxel_centers(self.mask[1], self.mask[0].shape)[self.mask[0] > 0]
        self.center, self.extent = pts.mean(0), np.ptp(pts, 0)

    def sample(self, rng):
        """A random slab: image (moving), affine (its header, with the start error), truth (true T1 mm per voxel),
        brain and valid (loss mask = both), jacobian, params."""
        params = draw_params(rng, self.center, self.extent)
        return {**make_sample(self.epi, self.field, self.mask, params, rng), 'params': params}
