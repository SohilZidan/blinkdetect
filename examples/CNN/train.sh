#!/bin/bash

: <<DESC
run the training steps for the CNN:
1. Sequence Building
2. Model Training
DESC

if [ $# -eq 0 ]; then
    echo "Dataset name is not set, please input one of the following:\n(BlinkingValidationSetVideos RN talkingFace eyeblink8 zju)"
    exit 1
fi

export PYTHONPATH=$HOME/blinkdetection

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

OUTPUT_FOLDER=${SCRIPT_DIR}/../../dataset/augmented_signals/versions
DATASET_NAME=$1
SUFFIX=vtest
PREFIX=archtest

# STEP="SEQUENCE BUILDING"
# echo "${STEP}..."
# python3 ${SCRIPT_DIR}/../../training/data_preparation.py \
#     --output_folder ${OUTPUT_FOLDER} \
#     --suffix ${SUFFIX} \
#     --overlap 10 \
#     --face_found \
#     --equal \
#     --dataset ${DATASET_NAME} \
#     --generate_plots
#     # --yaw_range -45 45 \
#     # --pitch_range -30 30 \
# if [ $? -ne 0 ]; then
#     echo "halt at ${STEP}"
#     exit 1
# fi

STEP="MODEL TRAINING"
echo "${STEP}..."
python3 ${SCRIPT_DIR}/../../training/train_blinkdetector.py \
    --annotation_file ${OUTPUT_FOLDER}/${DATASET_NAME}/annotations-${SUFFIX}.json \
    --dataset_path ${OUTPUT_FOLDER}/${DATASET_NAME}/${SUFFIX}/training \
    --prefix ${PREFIX} \
    --channels 1C  \
    --epoch 50 \
    --batch 4
if [ $? -ne 0 ]; then
    echo "halt at ${STEP}"
else
    echo "Training succeeded!!"
fi