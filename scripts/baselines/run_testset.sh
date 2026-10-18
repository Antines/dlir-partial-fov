#!/bin/bash
# Baselines on HCP: per subject, the first rest run that passes the QC (slabsim.dataset.passes_qc) -> real-distortion
# slabs (GRID pilot: 18, val: 36, full: 128 per run) -> all baselines (results.json per slab).
# Usage: [GRID=val|full] [DLIR_METHODS=...] bash run_testset.sh <subjects.txt> <prep_root> <testset_root> [parallel_jobs]
set -u
SUBJECTS=$1; PREP=$2; OUT=$3; JOBS=${4:-3}; GRID=${GRID:-pilot}
PY=${PY:-python}
CODE=$(cd "$(dirname "$0")/.." && pwd)
export OMP_NUM_THREADS=4 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4      # several jobs share the CPU
export DLIR_METHODS=${DLIR_METHODS:-header,rigid_oracle,affine_oracle,fsl_bbr,epi_reg,ants_rigid_mi,bridged_bbr,afni_lpc}   # SynthMorph: GPU

one_subject() {
    s=$1
    for run in rfMRI_REST1_LR rfMRI_REST1_RL rfMRI_REST2_LR rfMRI_REST2_RL; do
        q=$PREP/$s/$run/acq.json
        # the run passes the QC (slabsim.dataset.passes_qc)
        if [ -f "$q" ] && $PY -c "import json,sys; sys.path.insert(0,'$CODE'); from slabsim.dataset import passes_qc; sys.exit(not passes_qc(json.load(open('$q'))))"; then
            $PY $CODE/slabsim/build_testset.py $PREP/$s/$run $PREP/$s/anat $OUT/${s}_$run $GRID > $OUT/${s}_build.log 2>&1
            $PY $CODE/baselines/run_baselines.py $PREP/$s/anat $OUT/${s}_$run/t* > $OUT/${s}_baselines.log 2>&1
            echo "$s $run done"
            return
        fi
    done
    echo "$s no run passed the anchor QC"
}
export -f one_subject; export PY CODE PREP OUT GRID
mkdir -p $OUT
xargs -P $JOBS -I{} bash -c 'one_subject {}' < $SUBJECTS
