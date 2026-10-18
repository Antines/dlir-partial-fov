"""Image-domain forward model of a (partial-FOV) EPI acquisition: rigid head motion + susceptibility distortion.

Signal from a true position is encoded along the phase-encoding (PE) axis at the index
    j_encoded = j_true + s * kappa * dB0 * TRT
with dB0 the off-resonance field (Hz), TRT the total readout time (s), s = +1 for PE polarity 'j' (BIDS)
and -1 for 'j-', kappa a field-strength/shim scaling factor (Jezzard & Balaban 1995). Intensity is divided
by the PE Jacobian d(j_encoded)/d(j_true) (signal pile-up / stretching). The field moves rigidly with the head.
"""
import numpy as np
from scipy.ndimage import map_coordinates

from .geometry import apply, crop_slices, voxel_centers


def sample(data, affine, points, order=1):
    """Interpolate an image (data, voxel->world affine) at world points (..., 3); zero outside the image."""
    ijk = apply(np.linalg.inv(affine), points)
    return map_coordinates(data, np.moveaxis(ijk, -1, 0), order=order, mode='constant', cval=0.0)


def pe_lines(affine, shape, pe_axis, step=0.25, margin=10):
    """Dense samples along every PE line of a grid: fractional PE index t (T,) and world points (L, T, 3)."""
    other = [a for a in range(3) if a != pe_axis]
    t = np.arange(-margin, shape[pe_axis] - 1 + margin + step / 2, step)
    a, b = np.meshgrid(np.arange(shape[other[0]]), np.arange(shape[other[1]]), indexing='ij')
    ijk = np.zeros((a.size, t.size, 3))
    ijk[..., other[0]] = a.reshape(-1, 1)
    ijk[..., other[1]] = b.reshape(-1, 1)
    ijk[..., pe_axis] = t
    return t, apply(affine, ijk)


def invert_lines(t, enc, targets):
    """Solve enc(t) = target on every line (rows of enc are made monotonic first).

    Returns t* (L, P) and a mask of targets that fall inside the encoded range of their line.
    """
    enc = np.maximum.accumulate(enc, axis=1)
    L, T = enc.shape
    span = max(enc.max(), targets.max()) - min(enc.min(), targets.min()) + 10
    off = np.arange(L)[:, None] * span                       # stack all lines into one sorted array
    flat = (enc + off).ravel()
    start = np.arange(L)[:, None] * T
    idx = np.searchsorted(flat, (targets[None, :] + off).ravel()).reshape(L, -1)
    idx = np.clip(idx, start + 1, start + T - 1)
    e0, e1 = flat[idx - 1] - off, flat[idx] - off
    w = np.clip((targets[None, :] - e0) / np.maximum(e1 - e0, 1e-9), 0, 1)
    k = idx - 1 - start
    inside = (targets[None, :] >= enc[:, :1]) & (targets[None, :] <= enc[:, -1:])
    return t[k] + w * (t[k + 1] - t[k]), inside


def solve_distortion(affine, shape, pe_axis, shift_vox, step=0.25, margin=10):
    """True (undistorted) world position of every voxel of a distorted EPI grid.

    shift_vox(points) returns the PE displacement (in voxels) of signal originating at world points (..., 3).
    Returns the true positions (*shape, 3) in the grid's own world space, the PE Jacobian and a validity mask
    (False where the inversion is ambiguous: signal pile-up/folding or outside the encoded range).
    """
    t, pts = pe_lines(affine, shape, pe_axis, step, margin)
    enc = t[None, :] + shift_vox(pts)
    t_star, inside = invert_lines(t, enc, np.arange(shape[pe_axis], dtype=float))
    jac_dense = np.gradient(enc, t, axis=1)
    jac = np.array([np.interp(ts, t, jd) for ts, jd in zip(t_star, jac_dense)])

    other = [a for a in range(3) if a != pe_axis]
    a, b = np.meshgrid(np.arange(shape[other[0]]), np.arange(shape[other[1]]), indexing='ij')
    ijk = np.zeros(t_star.shape + (3,))
    ijk[..., other[0]] = a.reshape(-1, 1)
    ijk[..., other[1]] = b.reshape(-1, 1)
    ijk[..., pe_axis] = t_star
    to_grid = lambda x: np.moveaxis(x.reshape(shape[other[0]], shape[other[1]], shape[pe_axis], *x.shape[2:]), 2, pe_axis)
    return to_grid(apply(affine, ijk)), to_grid(jac), to_grid(inside & (jac > 0.2))


def simulate_slab(obj, field, slab_affine, slab_shape, head_pose=np.eye(4), pe_axis=1, pe_sign=1, trt=0.05,
                  field_scale=1.0):
    """Regime B: distorted slab EPI from an undistorted EPI-contrast image and a field map in head space.

    obj, field: (data, affine) pairs in head space (= T1 world space); field in Hz.
    head_pose: rigid head -> scanner transform during the EPI (head motion relative to the T1).
    Returns the slab image, the ground truth (true head-space position of each slab voxel, mm),
    the PE Jacobian and the validity mask.
    """
    to_head = np.linalg.inv(head_pose)
    shift = lambda pts: pe_sign * field_scale * trt * sample(*field, apply(to_head, pts))
    q, jac, valid = solve_distortion(slab_affine, slab_shape, pe_axis, shift)
    truth = apply(to_head, q)
    image = np.clip(sample(*obj, truth, order=3), 0, None) / np.clip(jac, 0.2, 5.0)
    return image, truth, jac, valid


def unwarp(raw, affine, field, pe_axis, pe_sign, trt):
    """Distortion-corrected image on the same grid: pull the raw image from the encoded position, times the Jacobian.

    raw: distorted EPI data on grid `affine`; field: (data, affine) field map (Hz) in undistorted space.
    """
    q = voxel_centers(affine, raw.shape)
    shift = pe_sign * trt * sample(*field, q)                 # encoded PE offset (voxels) of each true position
    e = affine[:3, pe_axis]
    enc = apply(np.linalg.inv(affine), q + shift[..., None] * e)
    step = 0.5 * e                                            # half a voxel along PE for the Jacobian
    jac = 1 + pe_sign * trt * (sample(*field, q + step) - sample(*field, q - step))
    out = map_coordinates(raw, np.moveaxis(enc, -1, 0), order=3, mode='constant', cval=0.0)
    return np.clip(out, 0, None) * np.clip(jac, 0, None)


def real_slab_truth(field, epi_affine, epi_shape, k0, k1, pe_axis, pe_sign, trt, epi_to_t1):
    """Regime A: ground truth for slices k0..k1-1 of a real distorted EPI (e.g. an HCP SBRef).

    field: (data, affine) field map (Hz) in the undistorted EPI scanner space (topup convention).
    epi_to_t1: 4x4 world transform from undistorted EPI space to T1 world space (whole-brain anchor).
    Returns the slab affine, the ground truth (T1-space position of each slab voxel, mm), Jacobian, valid mask.
    """
    slab_affine = crop_slices(epi_affine, k0)
    shape = (epi_shape[0], epi_shape[1], k1 - k0)
    shift = lambda pts: pe_sign * trt * sample(*field, pts)
    q, jac, valid = solve_distortion(slab_affine, shape, pe_axis, shift)
    return slab_affine, apply(epi_to_t1, q), jac, valid
