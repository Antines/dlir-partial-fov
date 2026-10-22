# Rigid-stage inference (no ground truth)

`predict_rigid` in `scripts/dl/predict_rigid.py` estimates a proper six-degree-of-freedom world-to-world transform from one EPI slab and the corresponding T1, using the frozen negative-NCC optimizer and its original configuration. **It does not need the validation truth, rigid oracle, or a trained CNN.**

```bash
python scripts/dl/predict_rigid.py \
  --slab /path/to/slab.nii.gz \
  --t1 /path/to/T1.nii.gz \
  --t1-mask /path/to/T1_mask.nii.gz \
  --params /path/to/params.json \
  --output /path/to/rigid.txt \
  --config configs/rigid_refine.yaml
```

Output: `rigid.txt` (4×4 RAS-mm residual rigid matrix), `rigid.json` (corrected affine, unchanged `pe_axis`, `pe_sign`, `trt`, signed PE direction, objective diagnostics). `corrected_affine = rigid @ slab_affine`; the output must be tested in Danil's deformation and resampling code before end-to-end results are claimed. For an on-the-fly PyTorch stream, import `predict_rigid(slab, slab_affine, t1, t1_affine, t1_mask=...)` directly and cache predictions outside the training loop if practical.

This is **numerical optimization**, not a trained neural-network inference method. The validated 1,080 saved slabs are the validation split; they must not be used for Danil's deformable-model training. The archived CNN and separate training source are provided solely for reproducibility of the previous learned baseline.
