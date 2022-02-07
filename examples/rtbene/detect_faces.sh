#!/bin/bash

export PYTHONPATH=$HOME/blinkdetection:$HOME/rt_gene/rt_gene/src

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

DATASET=temporal_rtbene_retina
# remove
rm -rf ${HOME}/${DATASET}/rtbene_w_faces_HDF5.h5
if [ $? -ne 0 ]; then
    echo "deletion failed"
else
    echo "deletion completed"
fi

python3 ${SCRIPT_DIR}/../../scripts/rtbene/detect_faces.py \
    --annotations ${HOME}/temporal_rtbene_v1/rtbene_raw_HDF5.h5 \
    --output ${HOME}/${DATASET}/rtbene_w_faces_HDF5.h5