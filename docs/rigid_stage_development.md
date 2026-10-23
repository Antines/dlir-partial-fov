# Rigid-stage development benchmark

The evaluation uses the fixed 1,080-slab validation distribution saved as `dlir_data/test/`. This is **not** the held-out 7,552-slab final test. The first run fixes a stratified 36-case manifest on Google Drive. All methods must use precisely that manifest.

## Methods

- `initial_header`: scanner header, 6-DOF identity residual.
- `initial_spatial_fusion`: archived 4,000-step CNN checkpoint, without retraining.
- `afni_rigid_6dof`: separate AFNI `lpc+ZZ`, `-partial_coverage`, `-large_move`, `-rigid_body` benchmark adapted from the repository's 12-DOF baseline. Requires installed AFNI or the existing Docker image.
- `refined_header` and `refined_spatial_fusion`: optional multi-resolution PyTorch optimization, **not** equivalent to AFNI LPC+ZZ. Must first pass the objective discrimination gate on the *same* manifest.

All returned transforms map current slab RAS world coordinates to T1 RAS world coordinates. Integrate through `corrected_affine = rigid_matrix @ slab_affine`; preserve `pe_axis`, `pe_sign`, `trt`. No ground-truth information is used to construct any predicted transform or decide between starts. The rigid oracle appears **only** in diagnostic objective ranking and evaluation.

The 12-DOF AFNI baseline is kept untouched. A 6-DOF result cannot be inferred from its reported 1.551 mm validation TRE.

## Google Colab

Upload `rigid_stage_eval_bundle.zip` to `MyDrive/NeuroML_DLIR/` and open `notebooks/rigid_stage_development.ipynb`. All Colab code is in the notebook; existing historical notebooks should not be rerun.

The notebook restores the original repository ZIP and HCP archive only if necessary, installs the isolated patch, runs tests, authenticates online W&B, locks the validation manifest, evaluates the saved CNN and similarity oracle-ranking, and attempts AFNI only if an AFNI executable or Docker is available.

**AFNI is not preinstalled in a standard Colab environment.** If unavailable, the AFNI step is explicitly skipped and the summary contains only methods actually executed. Official installation instructions: https://afni.nimh.nih.gov/pub/dist/doc/htmldoc/background_install/install_instructs/steps_linux_ubuntu24.html . On a workstation where Docker already works, run the AFNI command below with `--backend docker`.

## Standalone AFNI rigid evaluation

```bash
python scripts/baselines/eval_afni_rigid_dev.py \
  --data /content/dlir_data \
  --manifest /content/drive/MyDrive/NeuroML_DLIR/outputs/rigid_dev_20261021/manifest.json \
  --output /content/drive/MyDrive/NeuroML_DLIR/outputs/rigid_dev_20261021/afni_rigid \
  --backend docker
```

## Similarity screening

```bash
python scripts/dl/eval_rigid_hybrid_dev.py \
  --data /content/dlir_data \
  --manifest /content/drive/MyDrive/NeuroML_DLIR/outputs/rigid_dev_20261021/manifest.json \
  --output /content/drive/MyDrive/NeuroML_DLIR/outputs/rigid_dev_20261021/cnn_similarity_screen \
  --checkpoint /content/drive/MyDrive/NeuroML_DLIR/outputs/rigid_spatial_fusion_easy_v1/best.pt \
  --mode score
```

Only if `oracle_preferred_pct >= 75` on this manifest is refinement allowed with `--mode refine --score-summary <score_output>/summary.json`. This threshold is a preliminary engineering gate, not proof the method works. The two refinement starts are reported separately, not selected per case by ground-truth TRE.

## Tracking and reproducibility

W&B project: `neuroml-dlir`; online authentication is mandatory before all tracked evaluation work. The run ID is saved in each output folder for reconnect/resume, and JSONL files are authoritative backups. Previously completed successful cases are skipped; missing/failed entries are retried. The manifest remains fixed even if later commands specify a different `--max-cases`.

The summary uses each case's **mean pointwise displacement** as TRE, then the median over cases. Failure is greater than 2 degrees of rotation or 2 mm centroid translation, matching the original evaluation. Tool failures are included as failures rather than silently dropped. Neither test-set leakage nor unverified AFNI metrics are permitted.

## Source and installation

Additive files only: no overwrites of `scripts/dl/rigid_model.py`, `train_rigid.py`, simulator, old experiments, source checkpoint, or baseline entry point.

Prerequisites: PyTorch, NumPy, SciPy, nibabel, PyYAML, W&B, pytest. AFNI separately required for its benchmark. On a fresh Colab kernel, mount Drive and extract the original archives first. Commit only source/config/notebook/tests/docs to GitHub; exclude large data, outputs, checkpoints and per-case anatomical images.
