#!/bin/bash
export PYTHONPATH=$HOME/blinkdetection:$HOME/rt_gene/rt_gene/src

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# remove existed output
rm -rf ${HOME}/temporal_rtbene_v1/rtbene_signals_HDF5.h5
if [ $? -ne 0 ]; then
    echo "deletion failed"
fi

python3 ${SCRIPT_DIR}/../../scripts/rtbene/build_signals.py \
    --annotations ${HOME}/temporal_rtbene_v1/rtbene_features_HDF5.h5 \
    --output ${HOME}/temporal_rtbene_v1/rtbene_signals_HDF5.h5 \
    --threshold 30 \
    --step 5