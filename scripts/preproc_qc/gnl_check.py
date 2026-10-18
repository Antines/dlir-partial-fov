"""Size of HCP gradient nonlinearity: raw T1w_MPR1 vs HCP's gradient-corrected T1w_acpc_dc_restore.

Rigid + smooth B-spline SyN (26 mm mesh: only scanner-scale warps); the nonlinear part's displacement inside the
head estimates the uncorrected gradient-nonlinearity distortion of the raw images.
Usage: python gnl_check.py <raw_hcp_dir> <prep_hcp_dir> <subject> [<subject> ...]
"""
import json
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ANTS = Path(sys.executable).parent


def run(tool, *args):
    subprocess.run([str(ANTS / tool)] + [str(a) for a in args], check=True, stdout=subprocess.DEVNULL)


if __name__ == '__main__':
    raw_root, prep_root = Path(sys.argv[1]), Path(sys.argv[2])
    for subject in sys.argv[3:]:
        out = prep_root / subject / 'gnl'
        out.mkdir(parents=True, exist_ok=True)
        fixed, moving = out / 'dc_1mm.nii.gz', out / 'raw_1mm.nii.gz'
        run('ResampleImage', 3, raw_root / subject / 'T1w' / 'T1w_acpc_dc_restore.nii.gz', fixed, '1x1x1', 0, 4)
        run('ResampleImage', 3, prep_root / subject / 'anat' / 'T1_n4.nii.gz', moving, '1x1x1', 0, 4)
        run('antsRegistration', '-d', 3, '-o', f'[{out / "gnl_"}]', '-r', f'[{fixed},{moving},1]',
            '-t', 'Rigid[0.1]', '-m', f'MI[{fixed},{moving},1,32,Regular,0.25]',
            '-c', '[1000x500x250,1e-7,10]', '-s', '3x2x1vox', '-f', '8x4x2',
            '-t', 'BSplineSyN[0.1,26,0,3]', '-m', f'CC[{fixed},{moving},1,4]',
            '-c', '[70x50x20,1e-7,10]', '-s', '2x1x0vox', '-f', '4x2x1')
        warp = nib.load(out / 'gnl_1Warp.nii.gz').get_fdata().squeeze()           # displacement (mm), fixed space
        dc = nib.load(fixed).get_fdata()
        head = dc > np.percentile(dc[dc > 0], 25)
        mag = np.linalg.norm(warp, axis=-1)[head]
        stats = {'mean_mm': float(mag.mean()), 'p95_mm': float(np.percentile(mag, 95)), 'max_mm': float(mag.max())}
        (out / 'gnl.json').write_text(json.dumps(stats, indent=1))
        print(subject, {k: round(v, 2) for k, v in stats.items()}, flush=True)
