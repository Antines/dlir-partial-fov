# Rigid-to-deformable interface

The release exports the **frozen header-initialized 6-DOF** rigid registration result per validation slab. It does not alter the slab intensity, reassign PE axes, or apply any deformation.

Each case contains `metadata.json` and `rigid.txt`, alongside a global `handoff.jsonl` and `handoff_summary.json`.

## Contract

- `rigid_matrix`: 4×4 proper rigid transform, current slab **RAS-mm world** → T1 **RAS-mm world**.
- `original_slab_affine`: 4×4 NIfTI slab voxel → scanner-world affine.
- `corrected_affine = rigid_matrix @ original_slab_affine`.
- `pe_axis`: original slab voxel-axis integer 0, 1, 2; it is **not** a T1 axis label.
- `pe_sign`: exactly the original ±1 acquisition sign.
- `trt`: original total readout time in seconds; unchanged.
- `pe_world_unit_vector`: corrected affine column `pe_axis`, normalized.
- `signed_pe_world_unit_vector`: `pe_sign * pe_world_unit_vector`.

When applying the deformable PE field in scanner coordinates, use the corrected voxel-to-world orientation and the original `pe_axis`, `pe_sign`, and `trt`. Do not rotate `pe_axis` as an integer or silently alter its sign. If the second stage works in another coordinate system or transform direction, explicitly convert its transform convention and verify known points before integrating.

```python
import json
import numpy as np
record = json.loads(open('cases/<group>/<slab>/metadata.json').read())
rigid = np.asarray(record['rigid_matrix'], dtype=np.float64)
slab_affine = np.asarray(record['original_slab_affine'], dtype=np.float64)
corrected_affine = rigid @ slab_affine
assert np.allclose(corrected_affine, record['corrected_affine'], atol=1e-5)
pe_axis, pe_sign, trt = record['pe_axis'], record['pe_sign'], record['trt']
```

The exporter rejects invalid transforms, invalid PE metadata, missing cases, missing transform files, and mismatches with an already exported case. It uses atomic metadata writes and produces resumable per-case output.

The handoff archive is **validation-only**. It contains no source volumes, dense physical truth, field maps or evaluation masks, and must not be represented as a held-out test result. It is prepared for the deformable-stage input adapter; a complete cascade requires running the deformable-stage implementation on representative cases and verifying the combined geometry.
