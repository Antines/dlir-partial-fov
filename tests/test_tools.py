"""Integration test: FSL / ANTs transforms converted to world matrices recover a known EPI -> T1 transform."""
import os
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

from slabsim.forward import sample
from slabsim.geometry import apply, rigid, voxel_centers
from slabsim.tools import ants_world, flirt_world, world_flirt

FSL = Path(os.environ['FSLDIR']) / 'share' / 'fsl' / 'bin'
ANTS = Path(sys.executable).parent


def phantom_t1():
    """Ellipsoids with different intensities on a 1 mm grid stored in LAS order (radiological)."""
    A = np.diag([-1.0, 1.0, 1.0, 1.0])
    A[:3, 3] = [80, -95, -70]
    x = voxel_centers(A, (161, 191, 141))
    data = np.zeros(x.shape[:3])
    for c, r, v in [((0, 0, 0), (60, 75, 55), 1.0), ((15, 10, 5), (20, 30, 15), 2.0),
                    ((-25, -20, 10), (15, 20, 25), 0.5), ((5, 40, -20), (25, 12, 10), 1.5)]:
        data[np.sum(((x - c) / r) ** 2, -1) < 1] = v
    return data, A


def test_flirt_and_ants_recover_known_transform(tmp_path):
    t1, t1_aff = phantom_t1()
    W = rigid(angles=(4, -3, 6), translation=(3, -4, 2))           # true EPI world -> T1 world
    epi_aff = np.diag([2.0, 2.0, 2.0, 1.0])
    epi_aff[:3, :3] = rigid(angles=(10, 0, 0))[:3, :3] @ np.diag([2.0, 2.0, 2.0])
    epi_aff[:3, 3] = [-90, -100, -60]
    epi = sample(t1, t1_aff, apply(W, voxel_centers(epi_aff, (90, 100, 70))), order=1)
    nib.save(nib.Nifti1Image(t1.astype(np.float32), t1_aff), tmp_path / 't1.nii.gz')
    nib.save(nib.Nifti1Image(epi.astype(np.float32), epi_aff), tmp_path / 'epi.nii.gz')
    brain = voxel_centers(epi_aff, epi.shape)[epi > 0]
    err = lambda T: np.linalg.norm(apply(T, brain) - apply(W, brain), axis=1).mean()

    # FLIRT started from the scanner header (world identity), as the baselines are
    world_flirt(np.eye(4), tmp_path / 'epi.nii.gz', tmp_path / 't1.nii.gz', tmp_path / 'init.mat')
    subprocess.run([FSL / 'flirt', '-in', tmp_path / 'epi.nii.gz', '-ref', tmp_path / 't1.nii.gz', '-dof', '6',
                    '-init', tmp_path / 'init.mat', '-nosearch', '-omat', tmp_path / 'flirt.mat'], check=True)
    assert err(flirt_world(tmp_path / 'flirt.mat', tmp_path / 'epi.nii.gz', tmp_path / 't1.nii.gz')) < 1.0  # convention bugs give >10 mm
    world_flirt(W, tmp_path / 'epi.nii.gz', tmp_path / 't1.nii.gz', tmp_path / 'true.mat')     # exact round trip
    assert np.allclose(flirt_world(tmp_path / 'true.mat', tmp_path / 'epi.nii.gz', tmp_path / 't1.nii.gz'), W, atol=1e-6)

    subprocess.run([ANTS / 'antsRegistration', '-d', '3', '-o', f'[{tmp_path}/ants_]', '-t', 'Rigid[0.1]',
                    '-m', f'MI[{tmp_path}/t1.nii.gz,{tmp_path}/epi.nii.gz,1,32,Regular,0.5]',
                    '-c', '[200x100x50,1e-7,10]', '-s', '2x1x0vox', '-f', '4x2x1'], check=True,
                   stdout=subprocess.DEVNULL)
    assert err(ants_world(tmp_path / 'ants_0GenericAffine.mat')) < 1.0
