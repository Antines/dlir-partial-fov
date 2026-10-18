"""Accuracy of an estimated slab -> T1 mapping against the simulated ground truth."""
import numpy as np

from .geometry import apply, voxel_centers


def displacement_error(estimate, truth, mask):
    """Euclidean error (mm) between estimated and true T1-space positions of the slab voxels in `mask`."""
    err = np.linalg.norm(estimate - truth, axis=-1)[mask]
    return {'mean': float(err.mean()), 'median': float(np.median(err)),
            'p95': float(np.percentile(err, 95)), 'max': float(err.max())}


def linear_estimate(world_epi_to_t1, slab_affine, slab_shape):
    """Estimated T1 position of every slab voxel for a linear (rigid/affine) world -> world transform.

    The header-only start (no registration) is linear_estimate(np.eye(4), ...).
    """
    return apply(world_epi_to_t1, voxel_centers(slab_affine, slab_shape))


def coordinate_maps(t1_affine, t1_shape):
    """x, y, z world-coordinate images on the T1 grid, shape (*t1_shape, 3).

    Resampling them into slab space with a tool's own T1 -> EPI resampling (flirt -applyxfm,
    antsApplyTransforms, 3dAllineate, mri_synthmorph apply) gives that tool's estimated T1 position of
    every slab voxel, independent of the tool's matrix conventions, and also for nonlinear warps.
    """
    return voxel_centers(t1_affine, t1_shape).astype(np.float32)


def fit_rigid(src, dst):
    """Least-squares rigid transform (Kabsch) mapping points src (N, 3) onto dst (N, 3), as a 4x4 matrix."""
    cs, cd = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((src - cs).T @ (dst - cd))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    T = np.eye(4)
    T[:3, :3] = Vt.T @ D @ U.T
    T[:3, 3] = cd - T[:3, :3] @ cs
    return T


def rigid_errors(estimate, truth, points, mask):
    """Rotation (deg) and translation (mm, at the slab centroid) error between the rigid parts of two mappings.

    points: scanner positions of the slab voxels; the rigid part of each mapping is its least-squares fit.
    """
    p = points[mask]
    E, G = fit_rigid(p, estimate[mask]), fit_rigid(p, truth[mask])
    R = E[:3, :3] @ G[:3, :3].T
    angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    c = p.mean(0)
    return {'rot_deg': float(angle), 'trans_mm': float(np.linalg.norm(apply(E, c) - apply(G, c)))}


def label_dice(labels, labels_affine, estimate, truth, mask, ids):
    """Dice per label between T1 labels pulled into the slab with the estimated and the true mapping."""
    from .forward import sample
    est, ref = [sample(labels, labels_affine, x, order=0)[mask] for x in (estimate, truth)]
    return {int(i): float(2 * np.sum((est == i) & (ref == i)) / max(np.sum(est == i) + np.sum(ref == i), 1)) for i in ids}
