#!/bin/bash

export PYTHONPATH=$HOME/blinkdetection:$HOME/rt_gene/rt_gene/src
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

DATASET=temporal_rush
SUBJECTs=( 35 37 38 40 42 47)

# complete intervals
# rm -rf ${HOME}/${DATASET}/rush_raw_HDF5.h5
# for s in ${SUBJECTs[@]}; do
#     python3 ${SCRIPT_DIR}/../../scripts/rtbene/extract_complete_intervals.py \
#         --subject $s \
#         --blink_annotations ${HOME}/blinkdetection/dataset/BlinkingValidationSetVideos/${s}/${s}.json \
#         --faces ${HOME}/blinkdetection/dataset/BlinkingValidationSetVideos/${s}/frames \
#         --output ${HOME}/${DATASET}/rush_raw_HDF5.h5 \
#         --threshold 30 \
#         --dataset rush
# done

# CUT FACES
# python3 ${SCRIPT_DIR}/../../scripts/rtbene/cut_faces_rush.py \
#     --annotations ${HOME}/${DATASET}/rush_raw_HDF5.h5 \
#     --output ${HOME}/${DATASET}/rush_cutfaces_HDF5.h5 \
#     --face_annotations ${SCRIPT_DIR}/../../dataset/BlinkingValidationSetVideos \


# detect faces
# rm -rf ${HOME}/${DATASET}/rush_w_faces_HDF5.h5
# python3 ${SCRIPT_DIR}/../../scripts/rtbene/detect_faces.py \
#     --annotations ${HOME}/${DATASET}/rush_cutfaces_HDF5.h5 \
#     --output ${HOME}/${DATASET}/rush_w_faces_HDF5.h5

# estimate headpose
# rm -rf ${HOME}/${DATASET}/rush_w_pose_HDF5.h5
# python3 ${SCRIPT_DIR}/../../scripts/rtbene/estimate_headpose.py \
#     --annotations ${HOME}/${DATASET}/rush_w_faces_HDF5.h5 \
#     --output ${HOME}/${DATASET}/rush_w_pose_HDF5.h5

# features
# rm -rf ${HOME}/${DATASET}/rush_features_HDF5.h5
# python3 ${SCRIPT_DIR}/../../scripts/rtbene/extract_frame_features.py \
#     --annotations ${HOME}/${DATASET}/rush_w_pose_HDF5.h5 \
#     --output ${HOME}/${DATASET}/rush_features_HDF5.h5

# signals
# rm -rf ${HOME}/${DATASET}/rush_signals_HDF5.h5
# python3 ${SCRIPT_DIR}/../../scripts/rtbene/build_signals.py \
#     --annotations ${HOME}/${DATASET}/rush_features_HDF5.h5 \
#     --output ${HOME}/${DATASET}/rush_signals_HDF5.h5 \
#     --threshold 30 \
#     --step 5

# evaluate
rm -rf ${HOME}/${DATASET}/eval_results
python3 ${SCRIPT_DIR}/../../scripts/rtbene/evaluate.py \
    --annotations ${HOME}/${DATASET}/rush_signals_HDF5.h5 \
    --model ${HOME}/temporal_rtbene_v1/best_model_eyelids.pkl \
    --output ${HOME}/${DATASET}/eval_results
    --false_clips
