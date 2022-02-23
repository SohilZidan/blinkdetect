#!/bin/bash

export PYTHONPATH=$HOME/blinkdetection:$HOME/rt_gene/rt_gene/src

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

DATASET=temporal_rtbene_v1
# remove
rm -rf ${HOME}/${DATASET}/rtbene_signals_1_HDF5.h5
if [ $? -ne 0 ]; then
    echo "deletion failed"
fi

python3 ${SCRIPT_DIR}/../../scripts/rtbene/estimate_headpose.py \
    --annotations ${HOME}/${DATASET}/rtbene_signals_HDF5.h5 \
    --output ${HOME}/${DATASET}/rtbene_signals_1_HDF5.h5