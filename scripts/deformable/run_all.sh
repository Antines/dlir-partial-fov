#!/usr/bin/env bash
# All experiments of the deformable stage in order, then the two analyses. Run from scripts/ after `wandb login`.
# Data: dlir_data/ in the repository root or DLIR_DATA. RIGID (optional): CSV with the rigid model's transform of every
# validation slab (data.read_rigid), to also score each model on top of the rigid model.
set -e
for cfg in deformable/configs/*.yaml; do
    python -m deformable.train "$cfg" ${RIGID:+--rigid "$RIGID"}
done
python -m deformable.analysis mi-landscape
python -m deformable.analysis rigid-sensitivity ../runs/deformable/00_final ../runs/deformable/12_rigid_noise
