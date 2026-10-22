"""Multi-resolution, support-aware rigid EPI-to-T1 registration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class RefinementConfig:
    scales: tuple[float, ...] = (0.5, 1.0)
    steps: tuple[int, ...] = (35, 45)
    learning_rate: float = 0.10
    window: int = 7
    similarity: str = "negative_ncc"
    max_rotation_deg: float = 25.0
    max_translation_mm: float = 60.0
    minimum_valid_fraction: float = 0.30

    def __post_init__(self) -> None:
        if len(self.scales) != len(self.steps) or not self.scales:
            raise ValueError("scales and steps must have equal nonzero lengths")
        if any(s <= 0 or s > 1 for s in self.scales):
            raise ValueError("scales must lie in (0, 1]")
        if self.window % 2 != 1 or self.window < 3:
            raise ValueError("window must be an odd integer >= 3")
        if self.similarity not in {"negative_ncc", "absolute_ncc"}:
            raise ValueError("unsupported similarity")
        if any(step < 0 for step in self.steps):
            raise ValueError("steps must be nonnegative")
        if not 0 < self.minimum_valid_fraction <= 1:
            raise ValueError("minimum_valid_fraction must lie in (0, 1]")


def _volume_5d(x: Tensor) -> Tensor:
    if x.ndim != 4 or x.shape[0] != 1:
        raise ValueError(f"expected (1, X, Y, Z), got {tuple(x.shape)}")
    return x.permute(0, 3, 2, 1).unsqueeze(0).contiguous()


def _slab_level(slab: Tensor, affine: Tensor, scale: float) -> tuple[Tensor, Tensor]:
    nx, ny, nz = slab.shape[-3:]
    shape = (max(8, round(nx * scale)), max(8, round(ny * scale)), nz)
    volume = F.interpolate(
        _volume_5d(slab), size=(shape[2], shape[1], shape[0]),
        mode="trilinear", align_corners=True,
    )
    new_affine = affine.clone()
    spacing = torch.tensor(
        [(n - 1) / (m - 1) if m > 1 else 1.0 for n, m in zip((nx, ny, nz), shape)],
        device=affine.device, dtype=affine.dtype,
    )
    new_affine[:3, :3] = affine[:3, :3] @ torch.diag(spacing)
    return volume, new_affine


def _world_points(affine: Tensor, shape_zyx: Sequence[int]) -> Tensor:
    nz, ny, nx = shape_zyx
    z, y, x = torch.meshgrid(
        torch.arange(nz, device=affine.device, dtype=affine.dtype),
        torch.arange(ny, device=affine.device, dtype=affine.dtype),
        torch.arange(nx, device=affine.device, dtype=affine.dtype), indexing="ij",
    )
    ijk = torch.stack((x, y, z), dim=-1)
    return ijk @ affine[:3, :3].T + affine[:3, 3]


def _so3_exp(rotvec: Tensor) -> Tensor:
    x, y, z = rotvec.unbind()
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
    return torch.linalg.matrix_exp(skew)


def _increment(params: Tensor, center: Tensor) -> Tensor:
    rotation = _so3_exp(torch.deg2rad(params[:3]))
    translation = center - rotation @ center + params[3:]
    matrix = torch.eye(4, device=params.device, dtype=params.dtype)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def _sample_t1(t1: Tensor, t1_mask: Tensor, t1_affine: Tensor, world: Tensor) -> tuple[Tensor, Tensor]:
    nz, ny, nx = t1.shape[-3:]
    shape = (nx, ny, nz)
    inv = torch.linalg.inv(t1_affine)
    vox = world @ inv[:3, :3].T + inv[:3, 3]
    size = torch.tensor(shape, dtype=vox.dtype, device=vox.device)
    norm = 2.0 * vox / (size - 1) - 1.0
    grid = norm.unsqueeze(0)
    resampled = F.grid_sample(t1, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    support = ((vox >= 0) & (vox <= size - 1)).all(dim=-1)[None, None]
    brain = F.grid_sample(t1_mask, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return resampled, support.to(dtype=t1.dtype) * brain.clamp(0.0, 1.0)


def _local_correlation(a: Tensor, b: Tensor, weights: Tensor, window: int, similarity: str) -> Tensor:
    pool = lambda x: F.avg_pool3d(x, window, stride=1, padding=window // 2)
    count = pool(weights)
    denom = count.clamp_min(1e-5)
    mean_a = pool(weights * a) / denom
    mean_b = pool(weights * b) / denom
    var_a = pool(weights * a.square()) / denom - mean_a.square()
    var_b = pool(weights * b.square()) / denom - mean_b.square()
    covariance = pool(weights * a * b) / denom - mean_a * mean_b
    correlation = covariance / torch.sqrt((var_a.clamp_min(1e-5) * var_b.clamp_min(1e-5)))
    valid = (count >= 0.50) & (var_a > 1e-3) & (var_b > 1e-3)
    if similarity == "negative_ncc":
        score = correlation
    else:
        score = -torch.sqrt(correlation.square() + 1e-5)
    return torch.where(valid, score, 0.0).sum() / valid.sum().clamp_min(1)


def _prepare(slab: Tensor, t1: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    epi = _volume_5d(slab)
    anatomical = _volume_5d(t1)
    epi_valid = torch.isfinite(epi) & (epi > 0)
    if int(epi_valid.sum()) < 64:
        raise ValueError("slab contains fewer than 64 positive finite voxels")
    epi_values = epi[epi_valid]
    epi = (epi - epi_values.mean()) / epi_values.std(unbiased=False).clamp_min(1e-4)
    epi = torch.where(epi_valid, epi, 0.0)
    anatomical_valid = torch.isfinite(anatomical) & (anatomical > 0)
    anatomical_values = anatomical[anatomical_valid]
    if anatomical_values.numel() < 64:
        raise ValueError("T1 contains fewer than 64 positive finite voxels")
    anatomical = (anatomical - anatomical_values.mean()) / anatomical_values.std(unbiased=False).clamp_min(1e-4)
    anatomical = torch.where(anatomical_valid, anatomical, 0.0)
    return epi, anatomical, epi_valid.float()


def _level_loss(
    params: Tensor, initial: Tensor, center: Tensor, points: Tensor,
    epi: Tensor, t1: Tensor, t1_mask: Tensor, epi_valid: Tensor, t1_affine: Tensor,
    config: RefinementConfig,
) -> Tensor:
    matrix = _increment(params, center) @ initial
    world = points @ matrix[:3, :3].T + matrix[:3, 3]
    warped, support = _sample_t1(t1, t1_mask, t1_affine, world)
    weights = epi_valid * support
    valid_fraction = weights.sum() / epi_valid.sum().clamp_min(1)
    ncc = _local_correlation(epi, warped, weights, config.window, config.similarity)
    coverage_penalty = 2.0 * F.relu(config.minimum_valid_fraction - valid_fraction).square()
    return ncc + coverage_penalty


def refine_rigid(
    slab: Tensor, slab_affine: Tensor, t1: Tensor, t1_affine: Tensor,
    *, config: RefinementConfig = RefinementConfig(),
    initial_matrix: Tensor | None = None,
    t1_brain_mask: Tensor | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Return a 4x4 slab-world-to-T1-world residual rigid matrix.

    This is pairwise optimization, not a trained neural network. The optional
    initial matrix may be supplied by a separately trained rigid network.
    """
    device = slab.device
    dtype = torch.float32
    slab = slab.to(device=device, dtype=dtype)
    t1 = t1.to(device=device, dtype=dtype)
    slab_affine = slab_affine.to(device=device, dtype=dtype)
    t1_affine = t1_affine.to(device=device, dtype=dtype)
    initial = torch.eye(4, device=device, dtype=dtype) if initial_matrix is None else initial_matrix.detach().to(device=device, dtype=dtype)
    if initial.shape != (4, 4):
        raise ValueError("initial_matrix must be 4x4")
    voxel_center = torch.tensor([(n - 1) / 2 for n in slab.shape[-3:]], device=device, dtype=dtype)
    pivot = initial[:3, :3] @ (slab_affine[:3, :3] @ voxel_center + slab_affine[:3, 3]) + initial[:3, 3]
    factors = torch.tensor([5.0, 5.0, 5.0, 5.0, 5.0, 5.0], device=device, dtype=dtype)
    parameters = torch.nn.Parameter(torch.zeros(6, device=device, dtype=dtype))
    _, t1_5d, _ = _prepare(slab, t1)
    if t1_brain_mask is not None:
        if t1_brain_mask.shape != t1.shape:
            raise ValueError("t1_brain_mask shape must match t1")
        t1_mask = _volume_5d(t1_brain_mask.to(device=device, dtype=dtype).clamp(0, 1))
    else:
        t1_mask = _volume_5d(((t1 > 0) & torch.isfinite(t1)).to(dtype=dtype))
    optimizer = torch.optim.Adam([parameters], lr=config.learning_rate)
    first_score = None
    last_score = None

    for scale, steps in zip(config.scales, config.steps):
        slab_level, level_affine = _slab_level(slab, slab_affine, scale)
        epi_valid = (slab_level > 0).to(dtype=dtype)
        values = slab_level[epi_valid.bool()]
        if values.numel() < 64:
            raise ValueError("too few positive voxels at a pyramid level")
        epi = torch.where(epi_valid.bool(), (slab_level - values.mean()) / values.std(unbiased=False).clamp_min(1e-4), 0.0)
        points = _world_points(level_affine, epi.shape[-3:])
        center = pivot

        if scale == config.scales[-1]:
            with torch.no_grad():
                first_score = float(_level_loss(torch.zeros_like(parameters), initial, center, points, epi, t1_5d, t1_mask, epi_valid, t1_affine, config))
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = _level_loss(parameters * factors, initial, center, points, epi, t1_5d, t1_mask, epi_valid, t1_affine, config)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("non-finite refinement loss")
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                parameters[:3].clamp_(-config.max_rotation_deg / 5, config.max_rotation_deg / 5)
                parameters[3:].clamp_(-config.max_translation_mm / 5, config.max_translation_mm / 5)
        with torch.no_grad():
            last_score = float(_level_loss(parameters * factors, initial, center, points, epi, t1_5d, t1_mask, epi_valid, t1_affine, config))

    with torch.no_grad():
        accepted = last_score <= first_score
        final = (_increment(parameters * factors, pivot) @ initial if accepted else initial).detach()
    return final, {"initial_objective": first_score, "final_objective": last_score, "accepted": accepted}


def score_rigid_similarity(
    slab: Tensor, slab_affine: Tensor, t1: Tensor, t1_affine: Tensor,
    matrix: Tensor, *, config: RefinementConfig = RefinementConfig(),
    t1_brain_mask: Tensor | None = None,
) -> float:
    """Score one rigid transform using the same full-resolution refinement objective."""
    with torch.no_grad():
        device = slab.device
        slab = slab.to(device=device, dtype=torch.float32)
        t1 = t1.to(device=device, dtype=torch.float32)
        slab_affine = slab_affine.to(device=device, dtype=torch.float32)
        t1_affine = t1_affine.to(device=device, dtype=torch.float32)
        matrix = matrix.to(device=device, dtype=torch.float32)
        volume, affine = _slab_level(slab, slab_affine, config.scales[-1])
        valid = (volume > 0).float()
        values = volume[valid.bool()]
        if values.numel() < 64:
            raise ValueError("too few valid EPI voxels")
        epi = torch.where(valid.bool(), (volume - values.mean()) / values.std(unbiased=False).clamp_min(1e-4), 0.0)
        _, anatomical, _ = _prepare(slab, t1)
        if t1_brain_mask is not None:
            if t1_brain_mask.shape != t1.shape:
                raise ValueError("t1_brain_mask shape must match t1")
            anatomical_mask = _volume_5d(t1_brain_mask.to(device=device, dtype=torch.float32).clamp(0, 1))
        else:
            anatomical_mask = _volume_5d(((t1 > 0) & torch.isfinite(t1)).float())
        points = _world_points(affine, epi.shape[-3:])
        zero = torch.zeros(6, device=device, dtype=torch.float32)
        return float(_level_loss(zero, matrix, torch.zeros(3, device=device), points, epi, anatomical, anatomical_mask, valid, t1_affine, config))
