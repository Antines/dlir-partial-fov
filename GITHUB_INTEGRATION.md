# Rigid registration stage

## Overview

This contribution implements 6-DOF rigid EPI-to-T1 registration for partial-field-of-view acquisitions. Two implementations are provided:

- A trained dual-encoder spatial-fusion CNN baseline (`scripts/dl/rigid_model_spatial_fusion_v1.py`), with its training entry point and evaluation code.
- A header-initialized, support-aware multiresolution rigid optimizer (`scripts/dl/rigid_refine.py`). The optimizer provides the best reported validation result; it is **numerical optimization, not a trained neural-network prediction**.

Both produce proper 4 × 4 residual rigid transforms in RAS millimetres. Slab geometry is corrected by `corrected_affine = rigid_matrix @ slab_affine`. Acquisition metadata `pe_axis`, `pe_sign`, and `trt` are preserved for downstream distortion correction.

## Repository layout

- `scripts/dl/`: CNN architecture, training, inference, refinement, evaluation, reports and metadata export.
- `scripts/baselines/`: optional AFNI rigid comparison.
- `configs/`: rigid refinement and archived CNN experiment configuration.
- `notebooks/`: full validation and transform-export workflows.
- `tests/`: registration, geometry, evaluation and handoff tests.
- `docs/`: evaluation results, limitations, inference instructions and deformable-stage interface.

## Installation and tests

Install the dependencies in `requirements-rigid.txt` alongside the repository's existing requirements. From the repository root, run:

```bash
python -m pytest tests/test_rigid_dev.py tests/test_rigid_refine.py tests/test_rigid_inference.py tests/test_rigid_pipeline_integrity.py tests/test_rigid_stage_release.py -q
```

## Inference and integration

Predict a 6-DOF correction for a slab without using registration ground truth:

```bash
python scripts/dl/predict_rigid.py \
  --slab /path/to/slab.nii.gz \
  --t1 /path/to/T1.nii.gz \
  --t1-mask /path/to/T1_mask.nii.gz \
  --params /path/to/params.json \
  --output /path/to/rigid.txt \
  --config configs/rigid_refine.yaml
```

For transform conventions, acquisition metadata, and per-slab exports, see `docs/deformable_handoff.md`. The saved transform archive is for the **validation** split; independent training slabs require their own predicted transforms. Deformable-stage integration must be tested separately.

## Validation result

On 1,080 fixed validation slabs (30 subjects), header-initialized rigid refinement achieved:

| Metric | Value |
|---|---:|
| Median per-slab mean-voxel TRE | 1.622 mm |
| Median rigid rotation error | 0.621° |
| Median centroid translation error | 0.489 mm |
| Rigid-pose failure rate (>2° or >2 mm) | 9.44% |
| Cases below 2 mm TRE | 66.20% |
| Paired cases improved relative to header | 1,024 / 1,080 |

The separate trained CNN baseline achieved 5.143 mm median TRE. The held-out final test split was not used in method development. Difficult large-start cases remain, and some errors exceed 20 mm. See `docs/rigid_stage_results.md` for subgroup results and limitations.

W&B project: `neuroml-dlir`. Run identifiers, configuration snapshots, and full metrics should be retained with the original experiments; this repository does not contain credentials or private storage paths.

## Data and artifacts

The HCP volumes, cached simulation data, checkpoints, generated transforms, and W&B local run files are not versioned in GitHub. The full-cascade evaluation is not claimed in the rigid-only results.
