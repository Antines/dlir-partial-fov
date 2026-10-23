
"""6-DOF rigid EPI->T1 registration network with spatial feature fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _as_tensor(x, *, device, dtype=torch.float32) -> Tensor:
    if isinstance(x, Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def _volume_xyz_to_5d(x: Tensor) -> Tensor:
    """(1,X,Y,Z) or (X,Y,Z) -> (1,C,Z,Y,X)."""
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.ndim != 4:
        raise ValueError(f"expected (C,X,Y,Z), got {tuple(x.shape)}")
    return x.permute(0, 3, 2, 1).unsqueeze(0).contiguous()


def _normalise(x: Tensor, mask: Tensor | None = None, eps: float = 1e-5) -> Tensor:
    if mask is None:
        mask = torch.isfinite(x) & (x != 0)
    else:
        mask = mask.bool() & torch.isfinite(x)

    if mask.sum() < 8:
        return torch.zeros_like(x)

    vals = x[mask]
    mean = vals.mean()
    std = vals.std(unbiased=False).clamp_min(eps)

    y = (x - mean) / std
    return torch.where(mask, y, torch.zeros_like(y))


def _coarse_t1_world_grid(
    t1_affine: Tensor,
    t1_shape_xyz: Sequence[int],
    out_shape_xyz: Sequence[int],
    *,
    device,
    dtype,
) -> Tensor:
    nx, ny, nz = [int(v) for v in t1_shape_xyz]
    ox, oy, oz = [int(v) for v in out_shape_xyz]

    xs = torch.linspace(0, nx - 1, ox, device=device, dtype=dtype)
    ys = torch.linspace(0, ny - 1, oy, device=device, dtype=dtype)
    zs = torch.linspace(0, nz - 1, oz, device=device, dtype=dtype)

    i, j, k = torch.meshgrid(xs, ys, zs, indexing="ij")
    ijk = torch.stack((i, j, k), dim=-1)

    R = t1_affine[:3, :3]
    t = t1_affine[:3, 3]

    return ijk @ R.T + t


def _world_to_grid(
    world_xyz: Tensor,
    src_affine: Tensor,
    src_shape_xyz: Sequence[int],
) -> tuple[Tensor, Tensor]:

    inv = torch.linalg.inv(src_affine)

    flat = world_xyz.reshape(-1, 3)
    vox = flat @ inv[:3, :3].T + inv[:3, 3]

    sx, sy, sz = [int(v) for v in src_shape_xyz]

    def norm(v, n):
        if n <= 1:
            return torch.zeros_like(v)
        return 2.0 * v / (n - 1) - 1.0

    gx = norm(vox[:, 0], sx)
    gy = norm(vox[:, 1], sy)
    gz = norm(vox[:, 2], sz)

    inside = (
        (vox[:, 0] >= 0) & (vox[:, 0] <= sx - 1) &
        (vox[:, 1] >= 0) & (vox[:, 1] <= sy - 1) &
        (vox[:, 2] >= 0) & (vox[:, 2] <= sz - 1)
    )

    ox, oy, oz = world_xyz.shape[:3]

    grid_xyz = torch.stack((gx, gy, gz), dim=-1).reshape(ox, oy, oz, 3)

    grid = (
        grid_xyz
        .permute(2, 1, 0, 3)
        .unsqueeze(0)
        .contiguous()
    )

    mask = (
        inside
        .reshape(ox, oy, oz)
        .permute(2, 1, 0)
        .unsqueeze(0)
        .unsqueeze(0)
    )

    return grid, mask


def resample_slab_to_t1_grid(
    slab: Tensor,
    slab_affine: Tensor,
    t1_affine: Tensor,
    t1_shape_xyz: Sequence[int],
    out_shape_xyz: Sequence[int],
) -> tuple[Tensor, Tensor]:

    device = slab.device
    dtype = slab.dtype

    world = _coarse_t1_world_grid(
        t1_affine,
        t1_shape_xyz,
        out_shape_xyz,
        device=device,
        dtype=dtype,
    )

    src_shape = slab.shape[-3:]

    grid, support = _world_to_grid(
        world,
        slab_affine,
        src_shape,
    )

    src = _volume_xyz_to_5d(slab)

    out = F.grid_sample(
        src,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )

    return out, support.to(dtype=dtype)


def downsample_t1(
    t1: Tensor,
    out_shape_xyz: Sequence[int],
) -> Tensor:

    src = _volume_xyz_to_5d(t1)

    oz = int(out_shape_xyz[2])
    oy = int(out_shape_xyz[1])
    ox = int(out_shape_xyz[0])

    return F.interpolate(
        src,
        size=(oz, oy, ox),
        mode="trilinear",
        align_corners=True,
    )


def slab_center_world(
    slab_affine: Tensor,
    slab_shape_xyz: Sequence[int],
) -> Tensor:

    c = torch.tensor(
        [(n - 1) / 2 for n in slab_shape_xyz],
        device=slab_affine.device,
        dtype=slab_affine.dtype,
    )

    return slab_affine[:3, :3] @ c + slab_affine[:3, 3]


def euler_xyz_matrix_deg(angles_deg: Tensor) -> Tensor:

    a = torch.deg2rad(angles_deg)

    rx, ry, rz = a.unbind(-1)

    one = torch.ones_like(rx)
    zero = torch.zeros_like(rx)

    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)

    Rx = torch.stack(
        (one, zero, zero,
         zero, cx, -sx,
         zero, sx, cx),
        -1,
    ).reshape(-1, 3, 3)

    Ry = torch.stack(
        (cy, zero, sy,
         zero, one, zero,
         -sy, zero, cy),
        -1,
    ).reshape(-1, 3, 3)

    Rz = torch.stack(
        (cz, -sz, zero,
         sz, cz, zero,
         zero, zero, one),
        -1,
    ).reshape(-1, 3, 3)

    return Rz @ Ry @ Rx


def rigid_matrix_from_params(
    params: Tensor,
    center_world: Tensor,
) -> Tensor:

    single = params.ndim == 1

    if single:
        params = params.unsqueeze(0)

    if center_world.ndim == 1:
        center_world = center_world.unsqueeze(0).expand(
            params.shape[0], -1
        )

    R = euler_xyz_matrix_deg(params[:, :3])
    shift = params[:, 3:]

    t = (
        center_world
        - torch.bmm(R, center_world.unsqueeze(-1)).squeeze(-1)
        + shift
    )

    T = (
        torch.eye(
            4,
            device=params.device,
            dtype=params.dtype,
        )
        .unsqueeze(0)
        .repeat(params.shape[0], 1, 1)
    )

    T[:, :3, :3] = R
    T[:, :3, 3] = t

    return T[0] if single else T


def apply_matrix(T: Tensor, points: Tensor) -> Tensor:
    return points @ T[:3, :3].T + T[:3, 3]


class ConvBlock(nn.Module):

    def __init__(
        self,
        cin: int,
        cout: int,
        stride: int = 1,
    ):
        super().__init__()

        groups = min(8, cout)

        while cout % groups:
            groups -= 1

        self.net = nn.Sequential(
            nn.Conv3d(
                cin,
                cout,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, cout),
            nn.GELU(),

            nn.Conv3d(
                cout,
                cout,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, cout),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ModalityEncoder(nn.Module):
    """Keep spatial feature maps instead of pooling each modality separately."""

    def __init__(
        self,
        in_channels: int,
        base: int,
    ):
        super().__init__()

        self.stage1 = ConvBlock(
            in_channels,
            base,
            stride=2,
        )

        self.stage2 = ConvBlock(
            base,
            base * 2,
            stride=2,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        return x


class SpatialFusionEncoder(nn.Module):
    """Fuse EPI and T1 while spatial correspondence is still available."""

    def __init__(self, base: int):
        super().__init__()

        # Each modality has 2*base channels.
        # We concatenate:
        #   EPI features
        #   T1 features
        #   absolute feature difference
        # giving 6*base input channels.
        self.fusion = nn.Sequential(
            ConvBlock(
                base * 6,
                base * 4,
                stride=2,
            ),
            ConvBlock(
                base * 4,
                base * 6,
                stride=2,
            ),
        )

        # Preserve coarse spatial layout instead of reducing immediately to 1 voxel.
        self.pool = nn.AdaptiveAvgPool3d((2, 2, 2))

        self.out_features = base * 6 * 2 * 2 * 2

    def forward(
        self,
        epi_features: Tensor,
        t1_features: Tensor,
    ) -> Tensor:

        diff = torch.abs(epi_features - t1_features)

        x = torch.cat(
            (
                epi_features,
                t1_features,
                diff,
            ),
            dim=1,
        )

        x = self.fusion(x)

        x = self.pool(x)

        return x.flatten(1)


@dataclass
class RigidOutput:
    params: Tensor
    matrix: Tensor
    slab_grid: Tensor | None = None
    slab_mask: Tensor | None = None
    t1_grid: Tensor | None = None


class RigidRegistrationNet(nn.Module):
    """Spatial-fusion 6-DOF rigid regressor."""

    def __init__(
        self,
        grid_shape_xyz: Sequence[int] = (64, 80, 80),
        base_channels: int = 12,
        max_rotation_deg: float = 25.0,
        max_translation_mm: float = 60.0,
    ):
        super().__init__()

        self.grid_shape_xyz = tuple(
            int(v) for v in grid_shape_xyz
        )

        self.max_rotation_deg = float(
            max_rotation_deg
        )

        self.max_translation_mm = float(
            max_translation_mm
        )

        # EPI intensity + slab support mask.
        self.epi_encoder = ModalityEncoder(
            2,
            base_channels,
        )

        self.t1_encoder = ModalityEncoder(
            1,
            base_channels,
        )

        self.fusion_encoder = SpatialFusionEncoder(
            base_channels
        )

        f = self.fusion_encoder.out_features

        self.head = nn.Sequential(
            nn.Linear(f, 256),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(256, 128),
            nn.GELU(),

            nn.Linear(128, 6),
        )

        # Start from identity transform.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)


    def preprocess(
        self,
        slab: Tensor,
        slab_affine: Tensor,
        t1: Tensor,
        t1_affine: Tensor,
    ):

        slab_grid, support = resample_slab_to_t1_grid(
            slab,
            slab_affine,
            t1_affine,
            t1.shape[-3:],
            self.grid_shape_xyz,
        )

        t1_grid = downsample_t1(
            t1,
            self.grid_shape_xyz,
        )

        slab_grid = _normalise(
            slab_grid,
            support > 0.5,
        )

        t1_mask = t1_grid > 0

        t1_grid = _normalise(
            t1_grid,
            t1_mask,
        )

        return slab_grid, support, t1_grid


    def forward(
        self,
        slab: Tensor,
        slab_affine: Tensor,
        t1: Tensor,
        t1_affine: Tensor,
        *,
        return_grids: bool = False,
    ) -> RigidOutput:

        device = next(self.parameters()).device

        slab = _as_tensor(
            slab,
            device=device,
        )

        slab_affine = _as_tensor(
            slab_affine,
            device=device,
        )

        t1 = _as_tensor(
            t1,
            device=device,
        )

        t1_affine = _as_tensor(
            t1_affine,
            device=device,
        )

        slab_grid, support, t1_grid = self.preprocess(
            slab,
            slab_affine,
            t1,
            t1_affine,
        )

        epi_input = torch.cat(
            (
                slab_grid,
                support,
            ),
            dim=1,
        )

        epi_features = self.epi_encoder(
            epi_input
        )

        t1_features = self.t1_encoder(
            t1_grid
        )

        fused = self.fusion_encoder(
            epi_features,
            t1_features,
        )

        raw = self.head(fused)[0]

        params = torch.cat(
            (
                self.max_rotation_deg
                * torch.tanh(raw[:3]),

                self.max_translation_mm
                * torch.tanh(raw[3:]),
            )
        )

        center = slab_center_world(
            slab_affine,
            slab.shape[-3:],
        )

        matrix = rigid_matrix_from_params(
            params,
            center,
        )

        return RigidOutput(
            params=params,
            matrix=matrix,
            slab_grid=slab_grid if return_grids else None,
            slab_mask=support if return_grids else None,
            t1_grid=t1_grid if return_grids else None,
        )
