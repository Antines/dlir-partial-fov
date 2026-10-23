"""Error measure and loss terms of the deformable stage."""
import torch

from .model import identity_grid


def voxel_errors(u, b, M):
    """Distance (mm) between the predicted and the true T1 position of every scored voxel. The predicted position of
    slab voxel v is M @ H @ (v + u(v)): H = slab header, M = rigid transform, u = predicted shift (voxels)."""
    A = (M.double() @ b['H'].double()).to(u.device, torch.float32)
    p = (identity_grid(u.shape[2:], u.device) + u[0]).permute(1, 2, 3, 0) @ A[:3, :3].T + A[:3, 3]
    return torch.linalg.norm(p - b['truth'].to(u.device), dim=-1)[b['mask'].to(u.device)]


def smoothness(v):
    """Mean squared difference between neighbouring voxels of a field (diffusion regulariser)."""
    return (v.diff(dim=2).pow(2).mean() + v.diff(dim=3).pow(2).mean() + v.diff(dim=4).pow(2).mean()) / 3


def mutual_information(a, b, weight, bins=32, max_vox=30000, gen=None):
    """Mutual information of two images (1, 1, X, Y, Z) over the voxels with weight > 0.01 (at most `max_vox`, drawn
    with `gen`). Intensities are clipped to [0, 1] and spread over the bins with a Gaussian window, so the joint
    histogram, and with it the MI, is differentiable."""
    a, b, w = a.flatten(), b.flatten(), weight.flatten()
    idx = torch.nonzero(w > 0.01).squeeze(1)
    if idx.numel() > max_vox:
        idx = idx[torch.randperm(idx.numel(), generator=gen)[:max_vox].to(idx.device)]
    a, b, w = a[idx].clamp(0, 1), b[idx].clamp(0, 1), w[idx]
    centers = torch.linspace(0, 1, bins, device=a.device)
    pa = torch.exp(-(a[:, None] - centers) ** 2 / (2 / (bins - 1) ** 2))
    pb = torch.exp(-(b[:, None] - centers) ** 2 / (2 / (bins - 1) ** 2))
    pa, pb = pa / pa.sum(1, keepdim=True), pb / pb.sum(1, keepdim=True)
    pab = (pa * w[:, None]).T @ pb / w.sum()                                 # joint histogram (bins, bins)
    outer = pab.sum(1, keepdim=True) @ pab.sum(0, keepdim=True)              # product of the marginals
    return (pab * torch.log((pab + 1e-8) / (outer + 1e-8))).sum()
