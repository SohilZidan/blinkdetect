#!/bin/bash

export PYTHONPATH=$HOME/blinkdetection

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

if [ $# -eq 0 ]; then
    echo "Dataset name is not set, please input one of the following:\n(BlinkingValidationSetVideos RN talkingFace eyeblink8 zju)"
    exit 1
fi

DATASET=$1
# DATASETS=( BlinkingValidationSetVideos RN talkingFace eyeblink8 zju )
SUFFIX=vtest

MODELS_FOLDER_NAME=best_model
MODELS_FOLDER=${SCRIPT_DIR}/../../${MODELS_FOLDER_NAME}
MODEL_FILE_NAMES=$(ls -A ${MODELS_FOLDER})

OUTPUT_FOLDER=${SCRIPT_DIR}/../../dataset/augmented_signals/versions

# Stage 0: Preprocessing
${SCRIPT_DIR}/preprocessing.sh ${DATASET}
# Stage 1: Data Preparation
STEP="SEQUENCE BUILDING"
echo "${STEP}..."
python3 ${SCRIPT_DIR}/../../training/data_preparation.py \
    --output_folder ${OUTPUT_FOLDER} \
    --suffix ${SUFFIX}-eval \
    --overlap 10 \
    --face_found \
    --eval \
    --dataset ${DATASET} \
    --generate_plots
if [ $? -ne 0 ]; then
    echo "halt at ${STEP}"
    exit 1
fi

STEP="Evaluation"
echo "${STEP}:"
for MFN in ${MODEL_FILE_NAMES[@]}; do

    MODEL_PATH=${MODELS_FOLDER}/${MFN}

    echo "Model: ${MODEL_PATH}"

    python3 ${SCRIPT_DIR}/../../training/evaluate_softmax.py \
    --model ${MODEL_PATH} \
    --annotation_file ${OUTPUT_FOLDER}/$DATASET/annotations-${SUFFIX}-eval.json \
    --dataset_path ${OUTPUT_FOLDER}/$DATASET/${SUFFIX}-eval/training \
    --dataset $DATASET \
    --generate_fnfp_plots

done
