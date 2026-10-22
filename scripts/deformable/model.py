"""Network of the deformable stage: U-Net -> velocity field -> scaling and squaring -> shift of every slab voxel.

Fields are (1, 3, X, Y, Z) in slab voxels; channel c is the shift along tensor axis c.
Canonical orientation: before the network the slab is transposed and flipped so that its phase-encoding (PE) axis is
always tensor axis Y with positive polarity, so one network serves every PE direction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv3d(cin, cout, 3, padding=1)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.act(self.conv(x))


class UNet(nn.Module):
    """VoxelMorph-style 3D U-Net: 4 max-pool levels down, 4 trilinear levels up with skip connections.
    Channels 16-32-32-32 (encoder) and 32-32-32-32-16-16 (decoder), times `width`."""

    def __init__(self, in_ch, out_ch, width=1.0):
        super().__init__()
        enc = [round(c * width) for c in (16, 32, 32, 32)]
        dec = [round(c * width) for c in (32, 32, 32, 32, 16, 16)]
        self.enc = nn.ModuleList()
        cin = in_ch
        for c in enc:
            self.enc.append(ConvBlock(cin, c))
            cin = c
        self.bottleneck = ConvBlock(cin, dec[0])
        self.dec = nn.ModuleList()
        cin = dec[0]
        for i, skip in enumerate(reversed(enc)):
            self.dec.append(ConvBlock(cin + skip, dec[i + 1]))
            cin = dec[i + 1]
        self.final = ConvBlock(cin, dec[-1])
        self.head = nn.Conv3d(dec[-1], out_ch, 3, padding=1)
        nn.init.normal_(self.head.weight, 0, 1e-5)          # start from (almost) no shift
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        skips = []
        for blk in self.enc:
            x = blk(x)
            skips.append(x)
            x = F.max_pool3d(x, 2)
        x = self.bottleneck(x)
        for blk, s in zip(self.dec, reversed(skips)):
            x = F.interpolate(x, size=s.shape[2:], mode='trilinear', align_corners=False)
            x = blk(torch.cat([x, s], 1))
        return self.head(self.final(x))


def device():
    """CUDA if available, else Apple MPS, else CPU."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def build(cfg):
    """Input: slab and T1 (+ PE axis and sign as two constant channels without canonical orientation).
    Output: one channel (shift along PE) or three (free 3D shift)."""
    return UNet(2 if cfg['canonical'] else 4, 1 if cfg['output'] == 'pe' else 3, cfg['width'])


def identity_grid(shape, device):
    return torch.stack(torch.meshgrid(*[torch.arange(n, device=device, dtype=torch.float32) for n in shape],
                                      indexing='ij'), 0)                     # (3, X, Y, Z)


def warp(img, u, padding='zeros'):
    """img sampled at x + u(x), trilinear. img (1, C, X, Y, Z), u (1, 3, X, Y, Z) in voxels."""
    shape = u.shape[2:]
    loc = identity_grid(shape, u.device)[None] + u
    for c, n in enumerate(shape):                       # grid_sample wants [-1, 1] and the axis order (Z, Y, X)
        loc[:, c] = 2 * loc[:, c] / (n - 1) - 1
    grid = loc.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]
    return F.grid_sample(img, grid, mode='bilinear', padding_mode=padding, align_corners=True)


def integrate(v, steps=7):
    """Scaling and squaring: the shift after following the velocity v for unit time, as 2^steps composed small
    shifts; smooth v gives a smooth shift without folds."""
    u = v / 2**steps
    for _ in range(steps):
        u = u + warp(u, u, padding='border')
    return u


def jacobian_det(u):
    """Determinant of d(x + u)/dx (central differences) for u (1, 3, X, Y, Z); <= 0 where the mapping folds."""
    g = [torch.gradient(u[0, c], dim=(0, 1, 2)) for c in range(3)]
    J = [[g[c][k] + (1.0 if c == k else 0.0) for k in range(3)] for c in range(3)]
    return (J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1])
            - J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0])
            + J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]))


def to_canonical(x, pe_axis, pe_sign):
    """(1, C, X, Y, Z) on the slab grid -> PE along tensor axis Y, positive polarity."""
    if pe_axis == 0:
        x = x.transpose(2, 3)
    if pe_sign < 0:
        x = x.flip(3)
    return x


def from_canonical(u, pe_axis, pe_sign):
    """Shift field (1, 3, ...) from the canonical frame back to the slab grid (axes and vector components)."""
    if pe_sign < 0:
        u = u.flip(3)
        u = torch.cat([u[:, :1], -u[:, 1:2], u[:, 2:]], 1)
    if pe_axis == 0:
        u = u.transpose(2, 3)[:, [1, 0, 2]]
    return u


def predict(net, cfg, x, pe_axis, pe_sign):
    """x (2, X, Y, Z): slab and T1 on the slab grid -> shift u (1, 3, X, Y, Z) in slab voxels, and the network's
    field v (the velocity, or the shift itself without integration) that the smoothness penalty acts on."""
    x = x[None].to(next(net.parameters()).device)
    if cfg['canonical']:
        x = to_canonical(x, pe_axis, pe_sign)
    else:
        ones = torch.ones_like(x[:, :1])
        x = torch.cat([x, ones * (1.0 if pe_axis == 1 else -1.0), ones * float(pe_sign)], 1)
    shape = x.shape[2:]
    x = F.pad(x, [p for n in reversed(shape) for p in (0, (-n) % 16)])     # 4 pooling levels: sizes divisible by 16
    out = net(x)
    if cfg['output'] == 'pe':
        v = torch.zeros(1, 3, *x.shape[2:], device=x.device)
        v[:, 1 if cfg['canonical'] else pe_axis] = out[:, 0]
    else:
        v = out
    u = (integrate(v) if cfg['integrate'] else v)[:, :, :shape[0], :shape[1], :shape[2]]
    return (from_canonical(u, pe_axis, pe_sign) if cfg['canonical'] else u), v
