#!/bin/bash
export PYTHONPATH=$HOME/blinkdetection:$HOME/rt_gene/rt_gene/src

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"


SUBJECTs=( 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 )
# SUBJECTs=( 0 )

rm -rf ${HOME}/temporal_rtbene_v1/rtbene_raw_HDF5.h5

for s in ${SUBJECTs[@]}; do
    python3 ${SCRIPT_DIR}/../../scripts/rtbene/extract_complete_intervals.py \
        --subject `printf %03d $s` \
        --blink_annotations ${HOME}/rt-bene/s`printf %03d $s`_blink_labels.csv \
        --faces ${HOME}/rt-gene/s`printf %03d $s`_noglasses/natural/face \
        --output ${HOME}/temporal_rtbene_v1/rtbene_raw_HDF5.h5 \
        --threshold 30
done