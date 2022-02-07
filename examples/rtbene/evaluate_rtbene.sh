#!/bin/bash

export PYTHONPATH=$HOME/blinkdetection:$HOME/rt_gene/rt_gene/src

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
DATASET=temporal_rtbene_v1
# remove
rm -rf ${HOME}/${DATASET}/eval_results
if [ $? -ne 0 ]; then
    echo "deletion failed"
else
    echo "deletion completed"
fi

python3 ${SCRIPT_DIR}/../../scripts/rtbene/evaluate.py \
    --annotations ${HOME}/${DATASET}/rtbene_signals_1_HDF5.h5 \
    --model ${HOME}/${DATASET}/best_model_eyelids_all.pkl \
    --output ${HOME}/${DATASET}/eval_results_all
