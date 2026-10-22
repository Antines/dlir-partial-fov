"""Deformable stage on synthetic inputs (no data needed): canonical orientation, scaling and squaring, PE-only
output, error measure."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')
from deformable.losses import voxel_errors  # noqa: E402
from deformable.model import build, from_canonical, integrate, predict, to_canonical  # noqa: E402

PE = pytest.mark.parametrize('pe_axis, pe_sign', [(0, -1), (0, 1), (1, -1), (1, 1)])


@PE
def test_canonical_orientation(pe_axis, pe_sign):
    """The PE direction becomes +Y, and a +1 voxel canonical shift is a pe_sign voxel shift along pe_axis."""
    x = torch.zeros(1, 1, 8, 6, 4)
    x += pe_sign * torch.arange(x.shape[2 + pe_axis]).view([-1 if a == pe_axis else 1 for a in range(3)])
    xc = to_canonical(x, pe_axis, pe_sign)
    assert (xc.diff(dim=3) > 0).all()
    uc = torch.zeros(1, 3, *xc.shape[2:])
    uc[:, 1] = 1
    u = from_canonical(uc, pe_axis, pe_sign)
    assert u.shape[2:] == x.shape[2:]
    assert (u[:, pe_axis] == pe_sign).all() and (u[:, 1 - pe_axis] == 0).all() and (u[:, 2] == 0).all()


def test_scaling_and_squaring_of_a_constant_velocity():
    v = torch.zeros(1, 3, 8, 8, 8)
    v[:, 1] = 1.5
    assert torch.allclose(integrate(v), v, atol=1e-5)


@PE
def test_pe_only_output(pe_axis, pe_sign):
    cfg = dict(output='pe', canonical=True, integrate=True, width=0.5)
    torch.manual_seed(0)
    net = build(cfg)
    torch.nn.init.normal_(net.head.weight, 0, 0.1)                 # a visible shift instead of almost zero
    u, _ = predict(net, cfg, torch.rand(2, 20, 18, 6), pe_axis, pe_sign)
    assert u.shape == (1, 3, 20, 18, 6)
    assert u[:, pe_axis].abs().max() > 0 and (u[:, [c for c in range(3) if c != pe_axis]] == 0).all()


def test_error_measure():
    """Exact rigid transform (here H = M = identity): the true PE shift gives zero error, no shift gives its length."""
    shape = (6, 5, 4)
    shift = np.random.default_rng(0).uniform(-3, 3, shape).astype(np.float32)
    truth = np.stack(np.meshgrid(*map(np.arange, shape), indexing='ij'), -1).astype(np.float32)
    truth[..., 1] += shift
    b = {'truth': torch.from_numpy(truth), 'mask': torch.ones(shape, dtype=torch.bool), 'H': torch.eye(4)}
    u = torch.zeros(1, 3, *shape)
    u[:, 1] = torch.from_numpy(shift)
    assert voxel_errors(u, b, torch.eye(4)).max() < 1e-5
    assert torch.allclose(voxel_errors(torch.zeros_like(u), b, torch.eye(4)), torch.from_numpy(np.abs(shift)).flatten())
