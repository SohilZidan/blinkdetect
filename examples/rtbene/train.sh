#!/bin/bash

export PYTHONPATH=$HOME/blinkdetection:$HOME/rt_gene/rt_gene/src

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# remove
rm -rf ${HOME}/temporal_rtbene_v1/best_model.pkl
if [ $? -ne 0 ]; then
    echo "deletion failed"
else
    echo "deletion completed"
fi

python3 ${SCRIPT_DIR}/../../scripts/rtbene/train.py \
    --annotations ${HOME}/temporal_rtbene_v1/rtbene_signals_1_HDF5.h5 \
    --output_model ${HOME}/temporal_rtbene_v1/best_model_eyelids_all.pkl \
    --threshold -1