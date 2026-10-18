"""Rigid transforms and acquisition grids. All coordinates are world (scanner) millimetres, RAS."""
import numpy as np


def rotation(rx, ry, rz):
    """3x3 rotation from angles in degrees about x, y, z (applied in that order)."""
    ax, ay, az = np.deg2rad([rx, ry, rz])
    Rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    Ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    Rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def rigid(angles=(0, 0, 0), translation=(0, 0, 0), center=(0, 0, 0)):
    """4x4 rigid transform: rotation about `center` followed by a translation (mm)."""
    R, c = rotation(*angles), np.asarray(center, float)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = c - R @ c + np.asarray(translation, float)
    return T


def apply(T, points):
    """Apply a 4x4 transform to points of shape (..., 3)."""
    return points @ T[:3, :3].T + T[:3, 3]


def slab_grid(center, angles=(0, 0, 0), fov=(200, 200), voxel=(2.0, 2.0, 2.0), thickness=32.0):
    """Affine and shape of a slab grid: in-plane axes i, j and slice axis k (the slab normal).

    With angles = 0 the slab is axial: i = +x (right), j = +y (anterior), k = +z (superior).
    """
    shape = (int(round(fov[0] / voxel[0])), int(round(fov[1] / voxel[1])), max(1, int(round(thickness / voxel[2]))))
    A = np.eye(4)
    A[:3, :3] = rotation(*angles) @ np.diag(voxel)
    A[:3, 3] = np.asarray(center, float) - A[:3, :3] @ ((np.array(shape) - 1) / 2)
    return A, shape


def crop_slices(affine, k0):
    """Affine of a sub-volume that starts at slice k0 of the original grid (same voxels, shifted origin)."""
    A = affine.copy()
    A[:3, 3] = affine[:3, 3] + k0 * affine[:3, 2]
    return A


def voxel_centers(affine, shape):
    """World coordinates of all voxel centres, shape (*shape, 3)."""
    ijk = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing='ij'), axis=-1)
    return apply(affine, ijk.astype(float))


def random_rigid(rng, low, high, center=(0, 0, 0)):
    """Rigid transform: rotation of U(low, high) degrees about a random axis through `center`
    and a translation of U(low, high) mm in a random direction."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = np.deg2rad(rng.uniform(low, high))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K           # Rodrigues
    t = rng.normal(size=3)
    c = np.asarray(center, float)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = c - R @ c + t / np.linalg.norm(t) * rng.uniform(low, high)
    return T
