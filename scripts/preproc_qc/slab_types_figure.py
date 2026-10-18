"""Slab types on one subject's T1: the 18 slab geometries of the regime-A grid (axial and coronal x 16/32/48 mm x
30/50/70 %) and 9 random training slabs (regime B), each at its true rigid position. Writes slab_types.png (sagittal
mid-slice), slab_types.nii.gz (one T1-grid mask per slab, 4D) and slab_types.txt (what each volume is).
Usage: python slab_types_figure.py <package_root> <subject>_<run> <out_dir>    e.g. dlir_data 436239_rfMRI_REST1_LR out
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slabsim.dataset import Run
from slabsim.evaluation import fit_rigid
from slabsim.geometry import apply, voxel_centers
from slabsim.samples import load_sample


def footprint(affine, shape, truth, mask, t1):
    """T1-grid mask of the slab's field of view at its true rigid position."""
    W = fit_rigid(voxel_centers(affine, shape)[mask], truth[mask])
    ijk = apply(np.linalg.inv(W @ affine), voxel_centers(t1.affine, t1.shape))
    return np.all((ijk > -0.5) & (ijk < np.array(shape) - 0.5), axis=-1)


if __name__ == '__main__':
    root, subject_run, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    subject, run = subject_run.split('_', 1)
    t1 = nib.load(root / 'data' / subject / 'anat' / 'T1.nii.gz')
    masks, names = [], []
    for orientation, suffix in (('axial', ''), ('coronal', '_cor')):
        for thickness in (16, 32, 48):
            for position in (30, 50, 70):
                img, A, truth, m = load_sample(root / 'test' / subject_run / f't{thickness}_p{position}_small{suffix}')
                masks.append(footprint(A, img.shape, truth, m, t1))
                names.append(f'test grid: {orientation} {thickness} mm at {position} %')
    r, rng = Run(root / 'data' / subject / run), np.random.default_rng(0)
    for _ in range(9):
        s = r.sample(rng)
        masks.append(footprint(s['affine'], s['image'].shape, s['truth'], s['brain'] & s['valid'], t1))
        p = s['params']
        names.append(f"training: {p['orientation']} {p['thickness']:.0f} mm, tilt {' / '.join(f'{a:.0f}' for a in p['angles'])} deg")
    nib.save(nib.Nifti1Image(np.stack(masks, -1).astype(np.uint8), t1.affine), out / 'slab_types.nii.gz')
    (out / 'slab_types.txt').write_text(''.join(f'volume {i}: {n}\n' for i, n in enumerate(names)))

    T = t1.get_fdata()
    brain = nib.load(root / 'data' / subject / 'anat' / 'T1_mask.nii.gz').get_fdata() > 0
    x = int(round(np.argwhere(brain)[:, 0].mean()))                       # sagittal slice through the brain centre
    background = T[x].T
    lo, hi = np.percentile(background[background > 0], [1, 99.5])
    groups = [(f'axial, {t} mm', masks[i:i + 3]) for i, t in zip((0, 3, 6), (16, 32, 48))] + \
             [(f'coronal, {t} mm', masks[i:i + 3]) for i, t in zip((9, 12, 15), (16, 32, 48))] + \
             [('training examples', masks[18 + i:21 + i]) for i in (0, 3, 6)]
    fig, axes = plt.subplots(3, 3, figsize=(12, 12.5))
    for ax, (title, group) in zip(axes.flat, groups):
        ax.imshow(background, cmap='gray', vmin=lo, vmax=hi, origin='lower')
        for m in group:
            ax.imshow(np.ma.masked_where(~m[x].T, m[x].T), cmap='autumn', alpha=0.3, origin='lower')
            ax.contour(m[x].T, levels=[0.5], colors='red', linewidths=1)
        ax.set_title(title)
        ax.axis('off')
    axes[0, 0].text(0.02, 0.02, 'anterior →', transform=axes[0, 0].transAxes, color='w', fontsize=9)
    fig.suptitle(f'Slab types on subject {subject} (sagittal mid-slice). Rows 1-2: test/ grid, positions 30 / 50 / 70 % '
                 f'of the brain extent;\nrow 3: random training slabs (any thickness 16-60 mm, tilt, axial or coronal)',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out / 'slab_types.png', dpi=110)
    print('\n'.join(names))
