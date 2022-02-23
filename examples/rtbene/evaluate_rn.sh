#!/bin/bash

export PYTHONPATH=$HOME/blinkdetection:$HOME/rt_gene/rt_gene/src
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# .tag files
split="test"
db_type="rn30"
anns_files=$(find ${HOME}/blinkdetection/dataset/RN/ -type f | grep -E ${split}.*${db_type}.*\.tag$)
DATASET="temporal_rn_${split}_${db_type}"

# complete intervals
# rm -rf ${HOME}/${DATASET}/rn_raw_HDF5.h5
# for ann_file in ${anns_files[@]}; do
#     sub_path=$(dirname $ann_file)
#     faces_dir="${sub_path}/frames"
#     SUBJECT=$(basename $sub_path)
#     python3 ${SCRIPT_DIR}/../../scripts/rtbene/extract_complete_intervals.py \
#         --subject $SUBJECT \
#         --blink_annotations $ann_file \
#         --faces ${faces_dir} \
#         --output ${HOME}/${DATASET}/rn_raw_HDF5.h5 \
#         --threshold 30 \
#         --dataset rn
# done


# CUT FACES
# python3 ${SCRIPT_DIR}/../../scripts/rtbene/cut_faces_rn.py \
#     --annotations ${HOME}/${DATASET}/rn_raw_HDF5.h5 \
#     --output ${HOME}/${DATASET}/rn_cutfaces_HDF5.h5 \
#     --part "${split}/${db_type}"

# detect faces
# echo "FACE DETECTION..."
# echo "------------------------------------------------------"
# rm -rf ${HOME}/${DATASET}/rn_w_faces_HDF5.h5
# python3 ${SCRIPT_DIR}/../../scripts/rtbene/detect_faces.py \
#     --annotations ${HOME}/${DATASET}/rn_cutfaces_HDF5.h5 \
#     --output ${HOME}/${DATASET}/rn_w_faces_HDF5.h5

# estimate headpose
# echo "------------------------------------------------------"
# echo "HEAD POSE ESTIMATION..."
# rm -rf ${HOME}/${DATASET}/rn_w_pose_HDF5.h5
# python3 ${SCRIPT_DIR}/../../scripts/rtbene/estimate_headpose.py \
#     --annotations ${HOME}/${DATASET}/rn_w_faces_HDF5.h5 \
#     --output ${HOME}/${DATASET}/rn_w_pose_HDF5.h5

# features
# echo "------------------------------------------------------"
# echo "FEATURE EXTRACTION..."
# rm -rf ${HOME}/${DATASET}/rn_features_HDF5.h5
# python3 ${SCRIPT_DIR}/../../scripts/rtbene/extract_frame_features.py \
#     --annotations ${HOME}/${DATASET}/rn_w_pose_HDF5.h5 \
#     --output ${HOME}/${DATASET}/rn_features_HDF5.h5

# signals
# echo "------------------------------------------------------"
# echo "SIGNAL BUILDING..."
# rm -rf ${HOME}/${DATASET}/rn_signals_HDF5.h5
# python3 ${SCRIPT_DIR}/../../scripts/rtbene/build_signals.py \
#     --annotations ${HOME}/${DATASET}/rn_features_HDF5.h5 \
#     --output ${HOME}/${DATASET}/rn_signals_HDF5.h5 \
#     --threshold 30 \
#     --step 5

# evaluate
echo "------------------------------------------------------"
echo "EVALUATION..."
rm -rf ${HOME}/${DATASET}/eval_results
python3 ${SCRIPT_DIR}/../../scripts/rtbene/evaluate.py \
    --annotations ${HOME}/${DATASET}/rn_signals_HDF5.h5 \
    --model ${HOME}/temporal_rtbene_v1/best_model_eyelids.pkl \
    --output ${HOME}/${DATASET}/eval_results \
    --false_clips
