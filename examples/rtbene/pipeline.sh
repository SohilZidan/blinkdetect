#!/bin/bash
export PYTHONPATH=$HOME/blinkdetection
#:$HOME/rt_gene/rt_gene/src

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"


SUBJECTs=( 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 )
DIR = temporal_rtbene_v1

# 1. complete intervals
rm -rf ${HOME}/${DIR}/rtbene_raw_HDF5.h5
for s in ${SUBJECTs[@]}; do
    python3 ${SCRIPT_DIR}/../../scripts/rtbene/extract_complete_intervals.py \
        --subject `printf %03d $s` \
        --blink_annotations ${HOME}/rt-bene/s`printf %03d $s`_blink_labels.csv \
        --faces ${HOME}/rt-gene/s`printf %03d $s`_noglasses/natural/face \
        --output ${HOME}/${DIR}/rtbene_raw_HDF5.h5 \
        --threshold 30
done


# 2. faces

rm -rf ${HOME}/${DIR}/rtbene_w_faces_HDF5.h5
python3 ${SCRIPT_DIR}/../../scripts/rtbene/detect_faces.py \
    --annotations ${HOME}/${DIR}/rtbene_raw_HDF5.h5 \
    --output ${HOME}/${DIR}/rtbene_w_faces_HDF5.h5



# 3. feature extraction
rm -rf ${HOME}/${DIR}/rtbene_features_HDF5.h5
python3 ${SCRIPT_DIR}/../../scripts/rtbene/extract_frame_features.py \
    --annotations ${HOME}/${DIR}/rtbene_w_faces_HDF5.h5 \
    --output ${HOME}/${DIR}/rtbene_features_HDF5.h5



# 4. sequence building - 30 frames with 5 frames as stepsize
rm -rf ${HOME}/${DIR}/rtbene_signals_HDF5.h5
python3 ${SCRIPT_DIR}/../../scripts/rtbene/build_signals.py \
    --annotations ${HOME}/${DIR}/rtbene_features_HDF5.h5 \
    --output ${HOME}/${DIR}/rtbene_signals_HDF5.h5 \
    --threshold 30 \
    --step 5


# 5. train
rm -rf ${HOME}/${DIR}/best_model.pkl
python3 ${SCRIPT_DIR}/../../scripts/rtbene/train.py \
    --annotations ${HOME}/${DIR}/rtbene_signals_1_HDF5.h5 \
    --output_model ${HOME}/${DIR}/best_model_eyelids_all.pkl \
    --threshold -1 # all available