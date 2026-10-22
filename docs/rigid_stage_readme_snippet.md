## Rigid EPI-to-T1 registration

The rigid-stage contribution includes a trained 6-DOF spatial-fusion CNN baseline and a separate support-aware, coarse-to-fine, six-degree-of-freedom numerical rigid refinement (`scripts/dl/rigid_refine.py`). The latter is the selected validation method, **not** a trained network.

On Raphael’s fixed 1,080-slab validation set (30 subjects), header-initialized rigid refinement achieved **1.622 mm median TRE**, **0.621° median rotation error**, **0.489 mm median centroid translation error**, **9.44% rigid-pose failures**, and **66.20%** of cases below 2 mm TRE. The best trained CNN baseline achieved **5.143 mm median TRE**. The held-out test set was not used to tune the rigid method. See [`docs/rigid_stage_results.md`](docs/rigid_stage_results.md) and [`docs/deformable_handoff.md`](docs/deformable_handoff.md).

For inference, the rigid transform is a proper RAS-mm matrix and is applied as `corrected_affine = rigid_matrix @ slab_affine` while preserving `pe_axis`, `pe_sign`, and `trt`. The validation handoff package is stored privately in Google Drive, not versioned with the repository.
