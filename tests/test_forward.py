"""Checks of the forward model on synthetic phantoms (run: python -m pytest tests -q, from project/code)."""
import numpy as np

from slabsim.forward import sample, simulate_slab, real_slab_truth
from slabsim.geometry import rigid, slab_grid, voxel_centers, apply, crop_slices

HEAD = np.diag([2.0, 2.0, 2.0, 1.0])
HEAD[:3, 3] = -100                                        # 100^3 grid of 2 mm voxels centred at the origin


def phantom():
    x = voxel_centers(HEAD, (100, 100, 100))
    blobs = [((20, -10, 5), 15), ((-25, 15, -10), 20), ((0, 30, 20), 12)]
    data = sum(np.exp(-np.sum((x - np.array(c)) ** 2, -1) / (2 * s ** 2)) for c, s in blobs)
    return data, HEAD


def field(values):
    return np.broadcast_to(values, (100, 100, 100)).astype(float), HEAD


def test_identity():
    A, shape = slab_grid(center=(0, 0, 0), thickness=20)
    img, truth, jac, valid = simulate_slab(phantom(), field(0.0), A, shape)
    assert np.allclose(truth, voxel_centers(A, shape), atol=1e-6)
    assert np.allclose(jac, 1) and valid.all()
    assert np.allclose(img, np.clip(sample(*phantom(), voxel_centers(A, shape), order=3), 0, None), atol=1e-6)


def test_constant_field_shift():
    # 20 Hz x 0.05 s = 1 voxel shift along +j (anterior): a voxel shows signal from 2 mm posterior
    A, shape = slab_grid(center=(0, 0, 0), fov=(120, 120), voxel=(2, 2, 2), thickness=20)  # inside the field grid
    _, truth, _, valid = simulate_slab(phantom(), field(20.0), A, shape, pe_axis=1, pe_sign=1, trt=0.05)
    expected = voxel_centers(A, shape) - np.array([0, 2.0, 0])
    assert np.allclose(truth[valid], expected[valid], atol=1e-6)
    _, truth, _, valid = simulate_slab(phantom(), field(20.0), A, shape, pe_axis=1, pe_sign=-1, trt=0.05)
    assert np.allclose(truth[valid], (voxel_centers(A, shape) + [0, 2.0, 0])[valid], atol=1e-6)


def test_head_motion():
    A, shape = slab_grid(center=(5, -5, 10), angles=(15, 5, 0), thickness=30)
    pose = rigid(angles=(3, -2, 4), translation=(4, -2, 3), center=(0, 0, 10))
    _, truth, _, _ = simulate_slab(phantom(), field(0.0), A, shape, head_pose=pose)
    assert np.allclose(truth, apply(np.linalg.inv(pose), voxel_centers(A, shape)), atol=1e-6)


def test_linear_field_jacobian():
    # dB0/dy = 0.5 Hz/mm -> 1 Hz per 2 mm voxel; encoded index slope = 1 + trt * 1 Hz = 1.05
    y = voxel_centers(HEAD, (100, 100, 100))[..., 1]
    A, shape = slab_grid(center=(0, 0, 0), fov=(120, 120), thickness=20)
    _, _, jac, valid = simulate_slab(phantom(), (0.5 * y, HEAD), A, shape, trt=0.05)
    assert np.allclose(jac[valid], 1.05, atol=1e-3)


def test_encoding_consistency_random_field():
    # the truth must re-encode exactly onto the voxel grid: j_true + s * trt * dB0(true) = j
    rng = np.random.default_rng(0)
    x = voxel_centers(HEAD, (100, 100, 100))
    f = 60 * np.exp(-np.sum((x - [10, 40, -20]) ** 2, -1) / 400) + rng.normal(0, 1) * x[..., 2] * 0.2
    A, shape = slab_grid(center=(0, 20, -10), angles=(-20, 0, 0), thickness=24)
    pose = rigid(angles=(2, 0, -3), translation=(1, 2, -1))
    _, truth, _, valid = simulate_slab(phantom(), (f, HEAD), A, shape, head_pose=pose, pe_sign=-1, trt=0.06)
    j_true = apply(np.linalg.inv(A) @ pose, truth)[..., 1]
    j_enc = j_true - 0.06 * sample(f, HEAD, truth)
    assert valid.mean() > 0.95
    assert np.allclose(j_enc[valid], np.indices(shape)[1][valid], atol=0.02)


def test_real_slab_truth_identity():
    A = np.diag([2.0, 2.0, 2.0, 1.0])
    A[:3, 3] = [-104, -90, -72]
    affine, truth, _, valid = real_slab_truth(field(0.0), A, (104, 90, 72), 20, 36, 0, 1, 0.06, np.eye(4))
    assert np.allclose(affine, crop_slices(A, 20))
    assert np.allclose(truth, voxel_centers(affine, (104, 90, 16)), atol=1e-6) and valid.all()


def test_unwarp_inverts_forward_model():
    # distort with the forward model, then unwarp with the same field: the object is recovered
    from slabsim.forward import unwarp
    x = voxel_centers(HEAD, (100, 100, 100))
    f = 40 * np.exp(-np.sum((x - [0, 20, 0]) ** 2, -1) / 300)
    A, shape = slab_grid(center=(0, 0, 0), fov=(120, 120), thickness=30)
    distorted, _, jac, valid = simulate_slab(phantom(), (f, HEAD), A, shape, pe_axis=1, pe_sign=-1, trt=0.06)
    corrected = unwarp(distorted, A, (f, HEAD), pe_axis=1, pe_sign=-1, trt=0.06)
    reference = sample(*phantom(), voxel_centers(A, shape), order=3)
    inner = np.zeros(shape, bool)
    inner[10:-10, 10:-10, :] = True
    err = np.abs(corrected - reference)[inner & valid]
    assert err.max() < 0.02 * reference.max() and np.abs(jac - 1).max() > 0.1


def test_make_sample_smoke():
    from slabsim.sampling import draw_params, make_sample
    rng = np.random.default_rng(1)
    obj = phantom()
    params = draw_params(rng, brain_center=(0, 0, 0), brain_extent_mm=80)
    params['fov'] = [120, 120]                               # keep the slab inside the phantom's field grid
    s = make_sample(obj, field(10.0), (obj[0] > 0.1, HEAD), params, rng)
    assert s['image'].shape == s['brain'].shape == s['truth'].shape[:3] and np.isfinite(s['image']).all()
    assert s['brain'].any()



def test_draw_params_coronal_and_header():
    from slabsim.sampling import draw_params
    rng = np.random.default_rng(0)
    params = [draw_params(rng, (0, 0, 0), (140, 170, 120)) for _ in range(200)]
    coronal = [p for p in params if p['orientation'] == 'coronal']
    assert 60 < len(coronal) < 140 and all(abs(rigid(p['angles'])[1, 2]) > 0.8 for p in coronal)   # normal along y
    assert max(np.linalg.norm(np.array(p['header_pose'])[:3, 3]) for p in params) > 10

def test_rigid_errors():
    from slabsim.evaluation import rigid_errors
    A, shape = slab_grid(center=(0, 0, 0), fov=(60, 60), thickness=20)
    p = voxel_centers(A, shape)
    T = rigid(angles=(0, 0, 5), translation=(2, 0, 0))
    err = rigid_errors(apply(T, p), p, p, np.ones(shape, bool))
    assert abs(err['rot_deg'] - 5) < 1e-6 and abs(err['trans_mm'] - 2) < 1e-6


def test_slab_shim_removes_quadratic_field():
    from slabsim.sampling import slab_shim
    x = voxel_centers(HEAD, (100, 100, 100))
    quad = 30 + 0.2 * x[..., 1] + 0.004 * x[..., 2] ** 2                      # purely 2nd-order field
    A, shape = slab_grid(center=(0, 0, 0), fov=(120, 120), thickness=30)
    shimmed, _ = slab_shim((quad, HEAD), (np.ones((100, 100, 100)), HEAD), voxel_centers(A, shape), 1.0)
    assert np.abs(shimmed).max() < 1e-6 * np.abs(quad).max() + 1e-6
