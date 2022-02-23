#!/bin/bash

export PYTHONPATH=$HOME/blinkdetection:$HOME/rt_gene/rt_gene/src

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# remove
rm -rf ${HOME}/temporal_rtbene_v1/visualized_samples
if [ $? -ne 0 ]; then
    echo "deletion failed"
else
    echo "deletion completed"
fi

python3 ${SCRIPT_DIR}/../../scripts/rtbene/visualize_annotation.py \
    --annotations ${HOME}/temporal_rtbene_v1/rtbene_signals_HDF5.h5 \
    --output ${HOME}/temporal_rtbene_v1/visualized_samples \
    --samples 10

# 0: 12668
# 1: 174
# 2: 220
# 3: 192
# 4: 787
# 5: 6