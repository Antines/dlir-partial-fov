"""Random slab acquisitions (regime B, training) and one-call sample generation with ground truth."""
import numpy as np

from .appearance import apply_appearance, slab_profile
from .forward import real_slab_truth, sample, simulate_slab
from .geometry import apply, random_rigid, rigid, slab_grid, voxel_centers

# Parameter ranges (uniform unless noted). Effective distortion = field_scale * TRT: HCP 3T ~0.06 s,
# typical 3T protocols 0.025-0.05 s, 7T slabs (field x ~2.3, TRT 20-35 ms) ~0.045-0.08 s.
RANGES = {
    'thickness_mm': (16, 60),            # slab extent along its normal (~10-40 % of the brain)
    'inplane_mm': (1.6, 2.5),            # source data are 2 mm: finer grids add no detail
    'slice_mm': (1.6, 2.5),
    'fov_mm': (180, 220),
    'coronal_p': 0.5,                    # share of coronal slabs (normal along y; the test set is half coronal)
    'tilt_x_deg': (-30, 30),             # pitch: oblique slabs (e.g. along the Sylvian fissure or the hippocampus)
    'tilt_y_deg': (-10, 10),
    'tilt_z_deg': (-10, 10),
    'position': (0.15, 0.85),            # slab centre as a fraction of the brain extent along the normal
    'distortion_s': (0.02, 0.09),        # field_scale * TRT
    'motion_rot_sd_deg': 2.5,            # normal, clipped at 3 SD
    'motion_trans_sd_mm': 3.0,
    'ramp_mm': (0, 4),
    'bias': (0, 0.3),
    'gamma': (0.8, 1.25),
    'snr': (10, 60),
    'shim_fraction': (0, 1),             # share of the 2nd-order field removed by a slab-specific shim
    'header_offset': (0, 10),            # start error in the header only: U(0, 10) deg and mm, random axis, direction
    'header_far_p': 0.2,                 # plus, sometimes, a repositioned head (another session): HCP scanner headers
    'header_far_mm': (10, 30),           # are a median 7.7 mm and up to 57 mm off the anchor
}


def draw_params(rng, brain_center, brain_extent_mm):
    """One random slab acquisition: orientation and geometry, PE, distortion strength, head pose, start error in
    the header, appearance. brain_extent_mm: the brain's extent along x, y, z (or one number for all)."""
    u = lambda k: float(rng.uniform(*RANGES[k]))
    coronal = bool(rng.uniform() < RANGES['coronal_p'])
    angles = (u('tilt_x_deg') + (90 if coronal else 0), u('tilt_y_deg'), u('tilt_z_deg'))   # 90 deg about x: z -> y
    normal = rigid(angles)[:3, 2]
    offset = (u('position') - 0.5) * float(np.linalg.norm(normal * np.broadcast_to(np.asarray(brain_extent_mm, float), 3)))
    center = np.asarray(brain_center, float) + offset * normal
    rot = np.clip(rng.normal(0, RANGES['motion_rot_sd_deg'], 3), -3 * RANGES['motion_rot_sd_deg'], 3 * RANGES['motion_rot_sd_deg'])
    trans = np.clip(rng.normal(0, RANGES['motion_trans_sd_mm'], 3), -3 * RANGES['motion_trans_sd_mm'], 3 * RANGES['motion_trans_sd_mm'])
    inplane = u('inplane_mm')
    header = random_rigid(rng, *RANGES['header_offset'], center)
    if rng.uniform() < RANGES['header_far_p']:
        d = rng.normal(size=3)
        header[:3, 3] += d / np.linalg.norm(d) * u('header_far_mm')
    return {
        'orientation': 'coronal' if coronal else 'axial', 'center': center.tolist(), 'angles': angles,
        'fov': [u('fov_mm')] * 2, 'voxel': [inplane, inplane, u('slice_mm')], 'thickness': u('thickness_mm'),
        'pe_axis': int(rng.choice([0, 1], p=[0.5, 0.5] if coronal else [0.3, 0.7])),   # 1 = A-P (axial), S-I (coronal)
        'pe_sign': int(rng.choice([-1, 1])),
        'distortion_s': u('distortion_s'),
        'head_rot_deg': rot.tolist(), 'head_trans_mm': trans.tolist(),
        'ramp_mm': u('ramp_mm'), 'bias': u('bias'), 'gamma': u('gamma'), 'snr': u('snr'),
        'shim_fraction': u('shim_fraction'), 'header_pose': header.tolist(),
    }


def slab_shim(field, brain_mask, slab_points, fraction):
    """Field after a slab-specific shim: remove `fraction` of the 2nd-order polynomial fitted over the slab's brain.

    field, brain_mask: (data, affine) in head space; slab_points: head-space positions of the slab voxels.
    """
    pts = slab_points[sample(*brain_mask, slab_points, order=0) > 0.5]
    if fraction <= 0 or len(pts) < 20:
        return field
    c = pts.mean(0)
    basis = lambda x: np.stack([np.ones(len(x))] + [((x - c) / 50.0)[:, i] for i in range(3)] +
                               [((x - c) / 50.0)[:, i] * ((x - c) / 50.0)[:, j] for i in range(3) for j in range(i, 3)], 1)
    coef, *_ = np.linalg.lstsq(basis(pts), sample(*field, pts), rcond=None)
    grid = voxel_centers(field[1], field[0].shape).reshape(-1, 3)
    return field[0] - fraction * (basis(grid) @ coef).reshape(field[0].shape), field[1]


def make_sample(obj, field, brain_mask, params, rng):
    """Simulated slab with ground truth from undistorted EPI, field map (Hz) and brain mask in head (T1) space.

    obj, field, brain_mask: (data, affine). Returns dict with image, affine (the slab header: scanner position with the
    header start error, which leaves the truth unchanged, as a different head position during the T1 would), truth
    (head-space mm per voxel), brain (slab voxels inside the brain), valid (unambiguous inversion) and the Jacobian.
    """
    A, shape = slab_grid(params['center'], params['angles'], params['fov'], params['voxel'], params['thickness'])
    pose = rigid(params['head_rot_deg'], params['head_trans_mm'], center=params['center'])
    head_points = apply(np.linalg.inv(pose), voxel_centers(A, shape))           # undistorted slab region, head space
    field = slab_shim(field, brain_mask, head_points, params.get('shim_fraction', 0.0))
    image, truth, jac, valid = simulate_slab(obj, field, A, shape, head_pose=pose, pe_axis=params['pe_axis'],
                                             pe_sign=params['pe_sign'], trt=params['distortion_s'])
    brain = sample(*brain_mask, truth, order=0) > 0.5
    image = apply_appearance(image, params['voxel'][2], rng, params['ramp_mm'], params['bias'], params['gamma'],
                             params['snr'], brain)
    return {'image': image, 'affine': np.asarray(params.get('header_pose', np.eye(4))) @ A,
            'truth': truth.astype(np.float32), 'brain': brain, 'valid': valid, 'jacobian': jac.astype(np.float32)}


def make_real_sample(sbref, field, epi_to_t1, brain_mask, k0, k1, pe_sign, trt, extra_pose=None, ramp_mm=0.0, pe_axis=0):
    """Regime A (semi-real test data): slices k0..k1-1 of a real distorted SBRef with ground truth.

    sbref: (data, affine) raw distorted SBRef; field: (data, affine) field map (Hz) in the undistorted SBRef
    frame; epi_to_t1: whole-brain anchor; brain_mask: (data, affine) in T1 space; slabs are cut along the third axis.
    extra_pose: optional rigid transform applied to the slab header. This is equivalent to a different head
    position during the T1 (which has no relevant distortion), so the real distortion stays physically exact.
    """
    data, affine = sbref
    A, truth, jac, valid = real_slab_truth(field, affine, data.shape, k0, k1, pe_axis, pe_sign, trt, epi_to_t1)
    if extra_pose is not None:
        A = extra_pose @ A
    image = data[:, :, k0:k1] * slab_profile(k1 - k0, np.linalg.norm(affine[:3, 2]), ramp_mm)[None, None, :]
    brain = sample(*brain_mask, truth, order=0) > 0.5
    return {'image': image.astype(np.float32), 'affine': A, 'truth': truth.astype(np.float32), 'brain': brain,
            'valid': valid, 'jacobian': jac.astype(np.float32)}
