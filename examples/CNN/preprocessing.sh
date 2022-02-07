#!/bin/bash
: <<DESC
run the preprocessing pipeline for the CNN:
1. Face detection
2. Head Pose estimation
3. Face alignment
3. Face Tracking
4. Eye Landmarks estimation
5. Long Sequence Generation
NOTE: fix face alignment so its output is the same as Tracking
DESC

if [ $# -eq 0 ]; then
    echo "Dataset name is not set, please input one of the following:\n(BlinkingValidationSetVideos RN talkingFace eyeblink8 zju)"
    exit 1
fi

export PYTHONPATH=$HOME/blinkdetection


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SCRIPTS_DIR=${SCRIPT_DIR}/../../scripts
DATASET=$1


STEP="Face Detection"
echo "${STEP}.."
python3 ${SCRIPTS_DIR}/detect_faces.py \
    --dataset ${DATASET}
if [ $? -ne 0 ]; then
    echo "halt at ${STEP}"
    exit 1
fi

STEP="Head Pose Estimation"
echo "${STEP}.."
python3 ${SCRIPTS_DIR}/estimate_head_poses.py \
    --dataset ${DATASET}
if [ $? -ne 0 ]; then
    echo "halt at ${STEP}"
    exit 1
fi


STEP="Face Alignment"
echo "${STEP}.."
python3 ${SCRIPTS_DIR}/align_faces_annotations.py \
    --dataset ${DATASET}
if [ $? -ne 0 ]; then
    echo "halt at ${STEP}"
    exit 1
fi

STEP="Face Tracking"
echo "${STEP}.."
python3 ${SCRIPTS_DIR}/track_faces.py \
    --dataset ${DATASET} \
    --closest
if [ $? -ne 0 ]; then
    echo "halt at ${STEP}"
    exit 1
fi

STEP="Eye landmarks Estimation"
echo "${STEP}.."
python3 ${SCRIPTS_DIR}/predict_eye_landmarks.py \
    --dataset ${DATASET}
if [ $? -ne 0 ]; then
    echo "halt at ${STEP}"
    exit 1
fi

STEP="Signal Generation"
echo "${STEP}.."
python3 ${SCRIPTS_DIR}/generate_signals.py \
    --dataset ${DATASET} \
    --generate_plots

if [ $? -ne 0 ]; then
    echo "halt at ${STEP}"
else
    echo "Preprocessing succeeded!!"
fi