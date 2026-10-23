"""Losses for the rigid stage."""
from __future__ import annotations

import torch
from torch import Tensor

from .rigid_model_spatial_fusion_v1 import apply_matrix


def geometric_tre_loss(matrix: Tensor, points: Tensor, target_points: Tensor) -> Tensor:
    """Mean Euclidean displacement error in physical millimetres."""
    pred = apply_matrix(matrix, points)
    return torch.linalg.vector_norm(pred - target_points, dim=-1).mean()


def matrix_point_mse(matrix: Tensor, points: Tensor, target_points: Tensor) -> Tensor:
    pred = apply_matrix(matrix, points)
    return torch.mean((pred - target_points) ** 2)
