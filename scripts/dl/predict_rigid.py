"""Truth-free 6-DOF rigid inference for a single EPI slab and same-subject T1."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from dl.rigid_dev_common import assert_rigid
from dl.rigid_refine import RefinementConfig, refine_rigid
from dl.rigid_stage_handoff import atomic_json, corrected_geometry


def predict_rigid(slab: torch.Tensor, slab_affine: torch.Tensor,
                  t1: torch.Tensor, t1_affine: torch.Tensor, *,
                  t1_mask: torch.Tensor | None = None,
                  config: RefinementConfig = RefinementConfig()) -> tuple[np.ndarray, dict]:
    """Array-level interface: (1,X,Y,Z) tensors and 4x4 RAS-mm affines, no truth required."""
    if tuple(slab.shape[:1]) != (1,) or slab.ndim != 4 or t1.ndim != 4 or t1.shape[0] != 1:
        raise ValueError("slab and T1 must each have shape (1,X,Y,Z)")
    if slab_affine.shape != (4, 4) or t1_affine.shape != (4, 4):
        raise ValueError("both affines must be 4x4")
    matrix, diag = refine_rigid(slab, slab_affine, t1, t1_affine,
                                config=config, t1_brain_mask=t1_mask)
    return assert_rigid(matrix.detach().cpu().numpy()), diag


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--slab", type=Path, required=True)
    p.add_argument("--t1", type=Path, required=True)
    p.add_argument("--t1-mask", type=Path, required=True)
    p.add_argument("--params", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path, default=ROOT / "configs/rigid_refine.yaml")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    a = p.parse_args()
    import nibabel as nib
    dev = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else
                       "cpu" if a.device == "auto" else a.device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    imgs = [nib.load(str(q)) for q in (a.slab, a.t1, a.t1_mask)]
    if any(len(q.shape) != 3 for q in imgs):
        raise ValueError("Only 3-D NIfTI inputs are supported")
    slab_img, t1_img, mask_img = imgs
    if t1_img.shape != mask_img.shape or not np.allclose(t1_img.affine, mask_img.affine, atol=1e-3):
        raise ValueError("T1 brain mask and image grids must match")
    config = RefinementConfig(**yaml.safe_load(a.config.read_text())["refinement"])
    def tensor(img):
        return torch.from_numpy(img.get_fdata(dtype=np.float32)[None]).to(dev)
    matrix, diagnostics = predict_rigid(
        tensor(slab_img), torch.as_tensor(slab_img.affine, device=dev, dtype=torch.float32),
        tensor(t1_img), torch.as_tensor(t1_img.affine, device=dev, dtype=torch.float32),
        t1_mask=tensor(mask_img), config=config)
    params = json.loads(a.params.read_text())
    metadata = corrected_geometry(matrix, slab_img.affine,
                                  int(params["pe_axis"]), int(params["pe_sign"]), float(params["trt"]))
    metadata.update({"source_slab": str(a.slab), "source_t1": str(a.t1),
                     "method": "header-initialized numerical 6-DOF rigid refinement",
                     "diagnostics": diagnostics})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    temp = a.output.with_name(a.output.name + ".tmp")
    np.savetxt(temp, matrix, fmt="%.12g")
    os.replace(temp, a.output)
    atomic_json(a.output.with_suffix(".json"), metadata)
    print(json.dumps({"rigid": str(a.output), "metadata": str(a.output.with_suffix('.json')),
                      "accepted": diagnostics.get("accepted")}, indent=2), flush=True)

if __name__ == "__main__":
    main()
