import importlib.util
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from dl.predict_rigid import predict_rigid
from dl.rigid_refine import RefinementConfig


def test_truth_free_inference_returns_proper_rigid():
    vol = torch.zeros((1, 18, 20, 18), dtype=torch.float32)
    grid = torch.stack(torch.meshgrid(torch.arange(18),torch.arange(20),torch.arange(18),indexing='ij'),-1).float()
    vol[0] = (torch.exp(-((grid-9)**2).sum(-1)/28)+0.4*torch.exp(-((grid-torch.tensor([5.,7.,12.]))**2).sum(-1)/15))
    t1mask=(vol>0.01).float()
    matrix, diagnostic = predict_rigid(vol, torch.eye(4), vol.clone(),torch.eye(4),
         t1_mask=t1mask,config=RefinementConfig(scales=(1.0,),steps=(0,)))
    np.testing.assert_allclose(matrix,np.eye(4),atol=1e-5)
    assert np.linalg.det(matrix[:3,:3]) >0.999
    assert 'accepted' in diagnostic


def test_predict_rigid_cli_requires_original_acquisition_params():
    text = (ROOT/'scripts/dl/predict_rigid.py').read_text()
    assert '"--params"' in text and 'corrected_geometry' in text
    assert 'from dl.rigid_eval' not in text
    assert 'fit_rigid(' not in text
