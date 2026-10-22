from __future__ import annotations
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_rigid_pipeline_internal_dependencies_present():
    source = ROOT / 'scripts/dl/eval_rigid_hybrid_dev.py'
    assert source.is_file()
    tree = ast.parse(source.read_text())
    imported = {n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)}
    assert 'dl.rigid_data' in imported and 'dl.rigid_eval' in imported
    for name in ('rigid_data', 'rigid_eval', 'rigid_refine', 'rigid_dev_common',
                 'rigid_model_spatial_fusion_v1', 'train_rigid_spatial_fusion_v1', 'rigid_losses'):
        path = ROOT / 'scripts/dl' / (name + '.py')
        assert path.is_file(), path
        ast.parse(path.read_text())

def test_historical_cnn_config_has_required_fields():
    config = (ROOT / 'configs/rigid_spatial_fusion_easy_v1.yaml').read_text()
    assert 'steps_per_epoch: 500' in config
    assert 'epochs: 8' in config
    assert 'enabled: true' in config
