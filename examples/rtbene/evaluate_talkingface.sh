#!/bin/bash

export PYTHONPATH=$HOME/blinkdetection:$HOME/rt_gene/rt_gene/src
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

DATASET=temporal_talkingFace
SUBJECT=0

# complete intervals
# rm -rf ${HOME}/temporal_talkingFace/tf_raw_HDF5.h5
# python3 ${SCRIPT_DIR}/../../scripts/rtbene/extract_complete_intervals.py \
#     --subject $SUBJECT \
#     --blink_annotations ${HOME}/blinkdetection/dataset/talkingFace/0/talking.tag \
#     --faces ${HOME}/blinkdetection/dataset/talkingFace/0/frames \
#     --output ${HOME}/temporal_talkingFace/tf_raw_HDF5.h5 \
#     --threshold 30 \
#     --dataset talkingFace


# detect faces
# rm -rf ${HOME}/${DATASET}/tf_w_faces_HDF5.h5
# python3 ${SCRIPT_DIR}/../../scripts/rtbene/detect_faces.py \
#     --annotations ${HOME}/${DATASET}/tf_raw_HDF5.h5 \
#     --output ${HOME}/${DATASET}/tf_w_faces_HDF5.h5

# estimate headpose
# rm -rf ${HOME}/${DATASET}/tf_w_pose_HDF5.h5
# python3 ${SCRIPT_DIR}/../../scripts/rtbene/estimate_headpose.py \
#     --annotations ${HOME}/${DATASET}/tf_w_faces_HDF5.h5 \
#     --output ${HOME}/${DATASET}/tf_w_pose_HDF5.h5

# features
# rm -rf ${HOME}/${DATASET}/tf_features_HDF5.h5
# python3 ${SCRIPT_DIR}/../../scripts/rtbene/extract_frame_features.py \
#     --annotations ${HOME}/${DATASET}/tf_w_pose_HDF5.h5 \
#     --output ${HOME}/${DATASET}/tf_features_HDF5.h5

# signals
# rm -rf ${HOME}/${DATASET}/tf_signals_HDF5.h5
# python3 ${SCRIPT_DIR}/../../scripts/rtbene/build_signals.py \
#     --annotations ${HOME}/${DATASET}/tf_features_HDF5.h5 \
#     --output ${HOME}/${DATASET}/tf_signals_HDF5.h5 \
#     --threshold 30 \
#     --step 5

# evaluate
rm -rf ${HOME}/${DATASET}/eval_results
python3 ${SCRIPT_DIR}/../../scripts/rtbene/evaluate.py \
    --annotations ${HOME}/${DATASET}/tf_signals_HDF5.h5 \
    --model ${HOME}/temporal_rtbene_v1/best_model_eyelids.pkl \
    --output ${HOME}/${DATASET}/eval_results \
    --false_clips
