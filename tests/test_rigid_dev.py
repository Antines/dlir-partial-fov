from pathlib import Path
import importlib.util
import json
import sys
from types import ModuleType

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from dl.rigid_dev_common import assert_rigid, ensure_manifest, select_cases, summarize


def test_rotation_accepted():
    angle = np.deg2rad(17)
    matrix = np.eye(4)
    matrix[:2, :2] = [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    np.testing.assert_allclose(assert_rigid(matrix), matrix)


@pytest.mark.parametrize("linear", [np.diag([1.02, 1, 1]), np.diag([-1, 1, 1]), np.array([[1, .1, 0], [0, 1, 0], [0, 0, 1]])])
def test_invalid_rigid_rejected(linear):
    matrix = np.eye(4)
    matrix[:3, :3] = linear
    with pytest.raises(ValueError):
        assert_rigid(matrix)


def test_subject_diverse_selection_is_deterministic():
    cases = [{"relative_dir": f"test/sub{i}_rfMRI_X/t{j}", "subject": f"sub{i}", "group": f"sub{i}_rfMRI_X",
              "orientation": ("axial", "coronal")[j % 2], "thickness_mm": (16,32,48)[j % 3],
              "position": j % 3, "misalignment": ("small", "large")[j % 2]}
             for i in range(6) for j in range(6)]
    first = select_cases(cases, 16, seed=123)
    second = select_cases(cases, 16, seed=123)
    assert first == second
    assert len(set(x["relative_dir"] for x in first)) == 16
    assert len(set(x["subject"] for x in first)) > 1


def test_manifest_is_immutable_if_count_changes(tmp_path):
    for k in range(2):
        d = tmp_path / "test" / f"s{k}_rfMRI_A" / "t16"
        d.mkdir(parents=True)
        (d / "params.json").write_text(json.dumps({"orientation":"axial", "thickness_mm":16, "position":30, "misalignment":"small"}))
        (d / "slab.nii.gz").write_bytes(b"placeholder")
    path = tmp_path / "manifest.json"
    one = ensure_manifest(tmp_path, path, count=1)
    assert one == ensure_manifest(tmp_path, path, count=2)
    assert len(one) == 1


def test_failed_cases_counted_as_failures():
    report = summarize([
        {"case":"a", "method":"m", "status":"ok", "mean_tre_mm":1.5, "rot_deg":0.3,"trans_mm":0.4,"failure":False,"runtime_s":1},
        {"case":"b", "method":"m", "status":"error", "failure":True},
    ])
    assert report["m"]["failure_pct"] == 50
    assert report["m"]["tool_failure_pct"] == 50
    assert report["m"]["median_tre_mm"] == 1.5


def test_afni_rigid_matrix_and_command(tmp_path, monkeypatch):
    if "nibabel" not in sys.modules:
        monkeypatch.setitem(sys.modules, "nibabel", ModuleType("nibabel"))
    path = REPO / "scripts/baselines/eval_afni_rigid_dev.py"
    spec = importlib.util.spec_from_file_location("eval_afni_rigid_dev", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    f = tmp_path / "m.aff12.1D"
    np.savetxt(f, np.eye(4)[:3])
    np.testing.assert_allclose(module.afni_matrix(f), np.eye(4))
    monkeypatch.setattr(module.shutil, "which", lambda _: "/usr/local/bin/align_epi_anat.py")
    cmd = module.afni_command("native", tmp_path / "anat", tmp_path / "out", 2)
    assert "-rigid_body" in cmd
    assert "-partial_coverage" in cmd
    assert "-large_move" in cmd
    assert "lpc+ZZ" in cmd
    assert "-warp" not in cmd
