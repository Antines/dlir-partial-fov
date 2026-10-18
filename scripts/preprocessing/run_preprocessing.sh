#!/bin/bash
# Preprocess every subject in splits/ (validation, test, training order; finished subjects are skipped), then the
# gradient-nonlinearity residual of every run (an inclusion criterion). Needs FSL 6.0.7, the conda env
# (environment.yml) and Docker with freesurfer/synthstrip.
# Usage: bash run_preprocessing.sh <raw_hcp_dir> <prep_hcp_dir> [parallel_jobs]    (raw_hcp_dir from download_hcp.py)
set -u
RAW=$1; PREP=$2; JOBS=${3:-8}; PY=${PY:-python}
S=$(cd "$(dirname "$0")/.." && pwd); SPLITS=$(dirname $S)/splits
export OMP_NUM_THREADS=4 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=4      # several subjects share the CPU
mkdir -p $PREP
cat $SPLITS/hcp_val.txt $SPLITS/hcp_test.txt $SPLITS/hcp_train.txt | xargs -P $JOBS -I{} sh -c \
    "[ -f $PREP/{}/rfMRI_REST2_RL/acq.json ] || $PY $S/preprocessing/prep_subject.py hcp $RAW/{}/unprocessed/3T $PREP/{} > $PREP/{}.log 2>&1"
# gradient nonlinearity: raw vs HCP-corrected T1 for the first 10 validation subjects (download_hcp.py --gdc 10),
# averaged into the scanner-fixed group field, then the residual of every run -> acq.json
head -10 $SPLITS/hcp_val.txt | xargs $PY $S/preproc_qc/gnl_check.py $RAW $PREP
$PY $S/preproc_qc/gnl_group.py $PREP
