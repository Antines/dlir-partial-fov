# Rigid-stage validation — frozen v1

**Data:** 1,080 saved validation slabs from 30 HCP subjects; the directory `dlir_data/test/` is the *validation* archive, not the separate held-out test split.

**Method:** Header-initialized, support-aware, coarse-to-fine, 6-DOF rigid optimization using local negative NCC. This is a numerical registration method, **not a trained neural network**. Earlier spatial-fusion CNN results are a separate learned baseline. The original simulator, truth generation, and validation evaluator were not modified.

## Validation metrics

| Metric | Header | Frozen rigid refinement |
|---|---:|---:|
| Cases evaluated | 1,080 | 1,080 |
| Median mean-voxel TRE | 6.0923 mm | **1.6220 mm** |
| Median rigid rotation error | 4.5025° | **0.6211°** |
| Median centroid translation error | 4.5031 mm | **0.4885 mm** |
| Failure rate | 87.87% | **9.44%** |
| TRE below 2 mm | 2.13% | **66.20%** |
| Median optimizer time | — | **0.5768 s** (GPU, optimizer only) |

Relative median TRE reduction versus header: **73.38%**. Paired TRE improved on **1,024/1,080 (94.81%)** slabs. **102** cases failed the rigid-pose criterion: rotation error >2° or centroid translation error >2 mm. A registration that completes numerically can still fail geometrically.

TRE is *mean voxelwise physical displacement per slab*, aggregated by the *median across slabs*. The 2-mm TRE rate is a separate criterion from rigid-pose failure. The full cell includes file loading and output logging, so median GPU optimization time is not total wall-clock latency.

The previous spatial-fusion CNN achieved 5.143 mm median TRE on this validation archive. The stored 12-DOF AFNI `lpc+ZZ` affine result was 1.551 mm median TRE and 13.98% failure. The rigid oracle was 1.544 mm median TRE. The latter two results are references, **not direct equivalents to a 6-DOF model**.

## Robustness

Refinement p90 TRE: **3.196 mm**; p95: **8.635 mm**; worst: **26.315 mm**. The completed notebook found **78 cases >5 mm**, **50 >10 mm**, and **6 >20 mm**. The upper tail is substantial and must not be suppressed by reporting only the median.

| Subgroup | n | Median TRE | Rigid-pose failure |
|---|---:|---:|---:|
| Axial | 540 | 1.313 mm | 13.70% |
| Coronal | 540 | 1.752 mm | 5.19% |
| 16-mm | 360 | 1.613 mm | 14.44% |
| 32-mm | 360 | 1.579 mm | 6.94% |
| 48-mm | 360 | 1.645 mm | 6.94% |
| Large initial error | 540 | 1.689 mm | 15.93% |
| Small initial error | 540 | 1.556 mm | 2.96% |

Recorded worst cases include:

| Validation case | TRE |
|---|---:|
| `158035 / t48_p70_large` | 26.315 mm |
| `165638 / t48_p70_large_cor` | 26.111 mm |
| `622236 / t48_p70_large_cor` | 25.576 mm |
| `158035 / t32_p30_large` | 25.488 mm |
| `158035 / t48_p30_large` | 25.158 mm |

Subject `158035` appears repeatedly. The audit notebook produces case-level paired CSV, subject-level summaries, regressions and visual overlays so these failures can be investigated without changing v1 hyperparameters. The negative-NCC screening preferred the rigid oracle over the header in 1,079/1,080 cases; this does not guarantee optimizer convergence.

## Reproducibility and scope

- Frozen validation source: `/content/drive/MyDrive/NeuroML_DLIR/outputs/rigid_dev_20261021/full_validation_1080/`.
- Primary saved outputs: `header_refinement/cases.jsonl`, `header_refinement/transforms/`, `similarity_screen/cases.jsonl`, `manifest.json`, `paired_report/summary.json`.
- W&B project: `neuroml-dlir`; the original 1,080-case W&B evaluation runs are retained.
- The full validation used simulated limited-coverage slabs with measured-field-map distortion and real HCP EPI anatomy. No claim is made about independently acquired partial-FOV clinical scans.
- The held-out final test subjects remain untouched by the development workflow.
- The deformable-stage model and its actual output scores are **not** contained in the available rigid-stage repository snapshot; this release prepares a validated affine/PE metadata handoff, not a verified full cascade.

**Source of numerical results:** execution outputs in `rigid_stage_full_validation_1080 (1).ipynb` supplied on October 22, 2026. The final handoff notebook was subsequently executed in Colab; the exported archive passed an integrity check confirming 1,080 rigid.txt files, 1,080 metadata.json files, and two global manifests (2,162 files total). This does not establish full rigid-to-deformable integration.
