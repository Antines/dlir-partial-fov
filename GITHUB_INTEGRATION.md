# Rigid-stage contribution — GitHub integration

## Additive upload

This folder contains the rigid-stage source (including the archived CNN training dependencies), configuration, unit tests, clean Colab notebooks, 1,080-case validation documentation, and the rigid-to-deformable exporter. Copy the contents to the **root of the current team repository on a feature branch**, keeping the existing repository files and teammate changes intact. Confirm the current HEAD before copying; these files were checked against a repository ZIP snapshot dated 2026-10-21 and not against live GitHub `main`.

Proposed branch: `rigid-stage-6dof-final-v1`.

Use the source files under `scripts/dl/` and `configs/rigid_refine.yaml` as the frozen `rigid-header-refine-val1080-v1` method. Its best final transforms are computed by **numerical rigid optimization, not by the trained CNN**; `rigid_model_spatial_fusion_v1.py` is included only for checkpoint-compatible reproduction of the separately evaluated learned baseline.

## Local validation

From repository root in an environment with numpy, scipy, nibabel, matplotlib, torch, PyYAML, wandb, pytest:

```bash
python -m pytest tests/test_rigid_dev.py tests/test_rigid_refine.py tests/test_rigid_stage_release.py -q
```

AFNI is optional; its **6-DOF benchmark has not been run on the supplied validation data**. Avoid reporting it as an achieved result. Raphael's previously reported 12-DOF AFNI affine benchmark is a separate baseline.

## Reproduction

1. Place the original HCP preprocessed archive and parameters patch in Google Drive as described by the team repository; this GitHub patch does not contain any HCP images.
2. For the original full validation, use `notebooks/rigid_stage_full_validation_1080.ipynb` with the **frozen** `configs/rigid_refine.yaml`. Do not rerun it unless necessary; saved 1,080-case results are in Drive.
3. For reporting and handoff, use `notebooks/rigid_stage_final_handoff.ipynb`; it reads completed registration outputs and does **not** run the optimizer again.
4. Save/obtain the 2 MB `rigid_to_deformable_handoff_v1.zip` from `MyDrive/NeuroML_DLIR/outputs/rigid_dev_20261021/full_validation_1080/final_handoff_v1/` and share that archive privately with the team. It is **validation-only** and cannot replace rigid predictions on Danil's distinct training slabs.
5. Verify integration in Danil's live code using `corrected_affine = T_rigid @ original_slab_affine` and the unchanged `pe_axis`, `pe_sign`, `trt`. No combined cascade metric is claimed in this package.

## Results and experiment tracking

- Validation: 1,080 saved slabs from 30 validation HCP subjects, physically simulated partial-FOV. True held-out test set not used.
- Median per-slab mean-voxel TRE: **1.622 mm** (header **6.092 mm**).
- Median rigid rotation error **0.621°**; median centroid translation error **0.489 mm**.
- Rigid-pose failure **9.44%** (102/1,080). This is rotation error >2° OR centroid translation error >2 mm; it is not the TRE<2mm metric.
- Paired TRE improved on **1,024/1,080** cases; **66.20%** have TRE<2 mm.
- Important limitation: p95 TRE ~8.635 mm, worst ~26.315 mm; report outliers.
- W&B [full validation refinement](https://wandb.ai/salmanhkazim-skolkovo-institute-of-science-and-technology/neuroml-dlir/runs/9ca64smq), [similarity screen](https://wandb.ai/salmanhkazim-skolkovo-institute-of-science-and-technology/neuroml-dlir/runs/rdirtyeo), [handoff audit](https://wandb.ai/salmanhkazim-skolkovo-institute-of-science-and-technology/neuroml-dlir/runs/ciho56yu).

See `docs/rigid_stage_results.md` and `docs/deformable_handoff.md` for metric and coordinate conventions. The current validated rigid outputs are numerical optimization results, with a trained CNN only as a baseline. The archived baseline training source is `scripts/dl/train_rigid_spatial_fusion_v1.py`, with `configs/rigid_spatial_fusion_easy_v1.yaml`; the historical 4,000-step training must not be rerun for final handoff. The original run used `SlabStream` difficulty `easy` with 8 epochs x 500 steps, a 180-slab development subset, W&B online tracking, and the archived spatial-fusion CNN architecture. The deformable-stage neural network belongs to the other team member and must be evaluated separately.

## Excluded from GitHub

Do not commit: `dlir_data/`, `*.nii`, `*.nii.gz`, HCP downloads, `*.tar.gz`, checkpoint `*.pt`, W&B `wandb/` logs, API credentials, `/root/.netrc`, generated 1,080-case transform folders, or `rigid_to_deformable_handoff_v1.zip`. Do not commit the earlier 80-cell exploratory notebook with repeated `%%writefile` cells; use the two clean notebooks above.

## Suggested commit

`Add validated 6-DOF rigid refinement and deformable-stage handoff`

## Completeness check for this repository ZIP

The base repository contains **no `scripts/dl/` modules**. The earlier candidate ZIP omitted `rigid_data.py` and `rigid_eval.py`, which caused a runtime import failure despite component-level tests passing. This corrected update includes both files, `rigid_losses.py`, and the checkpoint-compatible training and model sources. Verify the evaluator imports in an environment with installed `nibabel`, `torch`, `PyYAML`, and `wandb` before running GPU inference.

The complete evaluation and handoff outputs are stored in Google Drive and W&B; GitHub includes reproducible code and compact reports, **not** the 1,080-case matrices or HCP files.
