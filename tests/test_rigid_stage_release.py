import json
import sys
from pathlib import Path

import types
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts' / 'dl'))
from audit_rigid_v1 import audit, group_stats
from rigid_stage_handoff import build_handoff, corrected_geometry, rigid_matrix
from rigid_stage_qc import resample_t1


def test_transform_composition_and_pe_direction():
    a = np.eye(4)
    a[:3, :3] = np.diag([2.0, -3.0, 4.0])
    a[:3, 3] = [10, -2, 1]
    theta = np.deg2rad(90)
    r = np.eye(4)
    r[:3, :3] = [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
    r[:3, 3] = [3, 5, 0]
    x = corrected_geometry(r, a, pe_axis=1, pe_sign=-1, trt=.05)
    corrected = np.asarray(x['corrected_affine'])
    np.testing.assert_allclose(corrected, r @ a)
    idx = np.array([1.4, 2.0, 4.1, 1.0])
    np.testing.assert_allclose(corrected @ idx, r @ (a @ idx))
    np.testing.assert_allclose(np.asarray(x['signed_pe_world_unit_vector']), [-1, 0, 0], atol=1e-7)
    assert x['pe_sign'] == -1 and x['pe_axis'] == 1 and x['trt'] == .05


@pytest.mark.parametrize('matrix', [np.diag([2, 1, 1, 1]), np.diag([-1, 1, 1, 1]), np.full((4, 4), np.nan)])
def test_invalid_matrix_rejected(matrix):
    with pytest.raises(ValueError):
        rigid_matrix(matrix)


@pytest.mark.parametrize('axis,sign,trt', [(4, 1, .05), (1, 0, .05), (1, 1, -.2)])
def test_invalid_pe_rejected(axis, sign, trt):
    with pytest.raises(ValueError):
        corrected_geometry(np.eye(4), np.eye(4), axis, sign, trt)


def test_audit_paired_and_validation_gate():
    cases = []
    h, r = {}, {}
    for i in range(3):
        c = f'test/{1000+i}_rfMRI_REST1_LR/t16_p30_large'
        cases.append({'relative_dir': c, 'subject': str(1000 + i)})
        h[c] = {'status': 'ok', 'mean_tre_mm': 7 + i}
        r[c] = {'status': 'ok', 'mean_tre_mm': 1 + i, 'rot_deg': .5, 'trans_mm': .6,
                'failure': False, 'runtime_s': .1}
    s, rows, groups = audit({'data_split': 'validation', 'cases': cases}, h, r)
    assert s['paired_improved_count'] == 3
    assert s['median_refined_tre_mm'] == 2
    assert len(groups) > 0
    with pytest.raises(ValueError):
        audit({'data_split': 'test', 'cases': cases}, h, r)
    with pytest.raises(ValueError):
        audit({'data_split': 'validation', 'cases': cases}, h, dict(list(r.items())[:-1]))


def test_export_and_resume(tmp_path, monkeypatch):
    case = 'test/123456_rfMRI_REST1_LR/t16_p30_large'
    d = tmp_path / 'data' / case
    d.mkdir(parents=True)
    affine = np.eye(4)
    affine[:3, 3] = [20, 30, 40]
    (d / 'slab.nii.gz').touch()
    monkeypatch.setitem(sys.modules, 'nibabel', types.SimpleNamespace(load=lambda _: types.SimpleNamespace(affine=affine, shape=(5, 6, 4))))
    (d / 'params.json').write_text(json.dumps({'pe_axis': 1, 'pe_sign': -1, 'trt': .05}))
    transform_root = tmp_path / 'transforms'
    transform = transform_root / '123456_rfMRI_REST1_LR' / 't16_p30_large' / 'header' / 'rigid.txt'
    transform.parent.mkdir(parents=True)
    a = np.eye(4)
    a[:3, 3] = [1, 2, 3]
    np.savetxt(transform, a)
    manifest = {'data_split': 'validation', 'cases': [{'relative_dir': case, 'subject': '123456'}]}
    report = {'status': 'ok'}
    output = tmp_path / 'output'
    for _ in range(2):
        result = build_handoff(tmp_path / 'data', manifest, {case: report}, transform_root, output)
        assert result['n_cases'] == 1
    meta = json.loads((output / 'cases/123456_rfMRI_REST1_LR/t16_p30_large/metadata.json').read_text())
    np.testing.assert_allclose(np.asarray(meta['corrected_affine']), a @ affine)
    assert len((output / 'handoff.jsonl').read_text().splitlines()) == 1
    a[:3, 3] = [10, 2, 3]
    np.savetxt(transform, a)
    with pytest.raises(ValueError, match='mismatches current source'):
        build_handoff(tmp_path / 'data', manifest, {case: report}, transform_root, output)


def test_resample_t1_identity():
    rng = np.random.default_rng(9)
    t1 = rng.random((10, 10, 5), dtype=np.float32)
    actual = resample_t1(t1, np.eye(4), np.eye(4), t1.shape)
    np.testing.assert_allclose(actual, t1, atol=1e-6)
