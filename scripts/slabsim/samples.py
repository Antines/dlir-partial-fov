"""Saving and loading simulated samples: slab image + ground truth as NIfTI, parameters as JSON."""
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def save_sample(s, out, params):
    """Write slab.nii.gz (moving image), truth.nii.gz (true T1-space mm of every voxel, 4D), masks and params."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(s['image'], s['affine']), out / 'slab.nii.gz')
    nib.save(nib.Nifti1Image(s['truth'], s['affine']), out / 'truth.nii.gz')
    nib.save(nib.Nifti1Image((s['brain'] & s['valid']).astype(np.uint8), s['affine']), out / 'eval_mask.nii.gz')
    nib.save(nib.Nifti1Image(s['jacobian'], s['affine']), out / 'jacobian.nii.gz')
    (out / 'params.json').write_text(json.dumps(params, indent=1))


def load_sample(path):
    """Slab image, affine, ground truth and evaluation mask of a saved sample."""
    path = Path(path)
    slab = nib.load(path / 'slab.nii.gz')
    truth = nib.load(path / 'truth.nii.gz').get_fdata(dtype=np.float32)
    mask = nib.load(path / 'eval_mask.nii.gz').get_fdata() > 0
    return slab.get_fdata(dtype=np.float32), slab.affine, truth, mask
