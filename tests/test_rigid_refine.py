import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from dl.rigid_refine import (
    RefinementConfig,
    _increment,
    _sample_t1,
    _world_points,
    _volume_5d,
    refine_rigid,
    score_rigid_similarity,
)


def _phantom(points):
    x, y, z = points.unbind(-1)
    peaks = (
        1.3 * torch.exp(-((x - 13) ** 2 + (y - 17) ** 2 + (z - 11) ** 2) / 70)
        + 0.7 * torch.exp(-((x - 24) ** 2 + (y - 10) ** 2 + (z - 9) ** 2) / 35)
        + 0.6 * torch.exp(-((x - 18) ** 2 + (y - 26) ** 2 + (z - 13) ** 2) / 45)
    )
    return peaks + 0.02


def _sample_pair():
    full_affine = torch.eye(4)
    full_points = _world_points(full_affine, (23, 37, 37))
    t1 = _phantom(full_points).permute(2, 1, 0).unsqueeze(0).contiguous()
    slab_affine = torch.eye(4)
    slab_affine[:3, 3] = torch.tensor([4.0, 5.0, 5.0])
    slab_points = _world_points(slab_affine, (11, 29, 29))
    center = slab_points[5, 14, 14]
    target = _increment(torch.tensor([0.0, 0.0, 5.0, 2.0, -1.0, 0.0]), center)
    warped_world = slab_points @ target[:3, :3].T + target[:3, 3]
    slab = (2.2 - _phantom(warped_world)).permute(2, 1, 0).unsqueeze(0).contiguous()
    return slab, slab_affine, t1, full_affine, target


def test_world_grid_identity_sample():
    values = torch.arange(4 * 5 * 6, dtype=torch.float32).reshape(1, 4, 5, 6) + 1
    vol = _volume_5d(values)
    points = _world_points(torch.eye(4), (6, 5, 4))
    sampled, inside = _sample_t1(vol, torch.ones_like(vol), torch.eye(4), points)
    assert torch.allclose(sampled, vol, atol=1e-4)
    assert torch.all(inside == 1)


def test_rotation_is_rigid_and_differentiable():
    pose = torch.tensor([3.0, -2.0, 4.0, 1.0, 2.0, -3.0], requires_grad=True)
    transform = _increment(pose, torch.tensor([11.0, 12.0, 13.0]))
    rotation = transform[:3, :3]
    assert torch.allclose(rotation.T @ rotation, torch.eye(3), atol=1e-5)
    assert torch.allclose(torch.linalg.det(rotation), torch.tensor(1.0), atol=1e-5)
    transform[:3, 3].square().sum().backward()
    assert pose.grad is not None and torch.isfinite(pose.grad).all()


def test_cross_modal_objective_prefers_true_pose():
    slab, affine, t1, t1_affine, target = _sample_pair()
    cfg = RefinementConfig(scales=(1.0,), steps=(0,), window=5)
    aligned = score_rigid_similarity(slab, affine, t1, t1_affine, target, config=cfg)
    unaligned = score_rigid_similarity(slab, affine, t1, t1_affine, torch.eye(4), config=cfg)
    assert aligned < unaligned, (aligned, unaligned)


def test_refinement_returns_valid_rigid_transform():
    slab, affine, t1, t1_affine, _ = _sample_pair()
    cfg = RefinementConfig(scales=(0.6, 1.0), steps=(2, 2), window=5)
    result, diagnostic = refine_rigid(slab, affine, t1, t1_affine, config=cfg)
    assert result.shape == (4, 4)
    assert torch.allclose(result[:3, :3].T @ result[:3, :3], torch.eye(3), atol=1e-4)
    assert diagnostic["final_objective"] == diagnostic["final_objective"]


def test_brain_mask_argument_does_not_change_identity_mapping():
    slab, affine, t1, t1_affine, target = _sample_pair()
    mask = torch.ones_like(t1)
    cfg = RefinementConfig(scales=(1.0,), steps=(0,), window=5)
    reference = score_rigid_similarity(slab, affine, t1, t1_affine, target, config=cfg)
    masked = score_rigid_similarity(
        slab, affine, t1, t1_affine, target, config=cfg, t1_brain_mask=mask
    )
    assert abs(reference - masked) < 1e-6


def test_refinement_recovers_phantom_rotation():
    slab, affine, t1, t1_affine, target = _sample_pair()
    cfg = RefinementConfig(scales=(0.6, 1.0), steps=(25, 25), window=5)
    estimated, report = refine_rigid(slab, affine, t1, t1_affine, config=cfg)
    relative = estimated[:3, :3] @ target[:3, :3].T
    radians = torch.acos(((torch.trace(relative) - 1) / 2).clamp(-1, 1))
    assert torch.rad2deg(radians) < 1.0
    assert report["accepted"]
