"""Conversions between tool-specific transform files and 4x4 world (scanner mm, RAS) transforms."""
import numpy as np


def flirt_world(mat, src, ref):
    """FSL FLIRT matrix file (src -> ref) -> 4x4 world transform."""
    from fsl.data.image import Image
    from fsl.transform import flirt
    return flirt.fromFlirt(np.loadtxt(mat), Image(str(src)), Image(str(ref)), 'world', 'world')


def world_flirt(world, src, ref, mat):
    """4x4 world transform (src -> ref) -> FSL FLIRT matrix file, e.g. to start FLIRT from the scanner header."""
    from fsl.data.image import Image
    from fsl.transform import flirt
    np.savetxt(mat, flirt.toFlirt(world, Image(str(src)), Image(str(ref)), 'world', 'world'), fmt='%.10f')


def ants_world(mat_file):
    """ANTs/ITK affine file from a registration with fixed = T1, moving = EPI -> 4x4 world EPI -> T1."""
    import ants
    tx = ants.read_transform(str(mat_file))
    A, t, c = np.reshape(tx.parameters[:9], (3, 3)), np.array(tx.parameters[9:]), np.array(tx.fixed_parameters)
    lps = np.eye(4)
    lps[:3, :3], lps[:3, 3] = A, t + c - A @ c          # ITK maps fixed (T1) points to moving (EPI) points, LPS
    flip = np.diag([-1, -1, 1, 1])
    return np.linalg.inv(flip @ lps @ flip)


def plumb(src, dst, voxel=None):
    """Resample an image onto an axis-aligned (non-oblique) grid covering the same world region.

    Some tools (AFNI by default) treat oblique datasets as plumb; feeding them plumb copies keeps world coordinates exact.
    """
    import nibabel as nib
    from .forward import sample
    from .geometry import apply, voxel_centers
    img = nib.load(str(src))
    data = img.get_fdata(dtype=np.float32)
    voxel = voxel or float(min(img.header.get_zooms()[:3]))
    corners = apply(img.affine, np.array([[i, j, k] for i in (0, data.shape[0] - 1)
                                          for j in (0, data.shape[1] - 1) for k in (0, data.shape[2] - 1)], float))
    lo, hi = corners.min(0), corners.max(0)
    shape = tuple(int(np.ceil((h - l) / voxel)) + 1 for l, h in zip(lo, hi))
    A = np.diag([voxel, voxel, voxel, 1.0])
    A[:3, 3] = lo
    out = sample(data, img.affine, voxel_centers(A, shape), order=1)
    nib.save(nib.Nifti1Image(out.astype(np.float32), A), str(dst))
