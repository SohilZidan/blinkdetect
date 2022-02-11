### requirements:
* python 3.6.9

### components and pipeline
1. face detection: using retina-face python package 0.0.4
2. face head pose estimation: 3DDFA_V2
3. face tracking: !! KLT and deepface
4. eye landmarks estimation: face mesh estimation: Mediapipe -- file eye_landmarks_predict append new information to the output of the face tracker
5. signals generation


<!-- ### TODO
* debugging code -- logging
* configs -->


### Building
* clone this repository: `git clone --recursive <project url>`
* `git submodule update --init --recursive`
* cd <project url>
* sh ./build.sh
* export PYTHONPATH=<path of the repo>

### Preprocessing Steps
1. generate frames of the videos
   * place videos inside <repo_root>/dataset/$DATASET_NAME
   * run generate_frames_v3.sh
   * the output structure of the videos is as follows:
      ```
      ├── $DATASET_NAME
      |   ├── video_1
      │   │   ├── video_1.ext
      │   │   ├── frames
      │   │   |   ├── 000000.png
      │   │   |   ├── 000001.png
      │   │   |   ├── ...
      │   ├── video_2
      │   │   ├── video_2.ext
      │   ├── ...
      │   └── video_n
      ├── ...
      └── ...
      ```
   
2. Face detection

   ```bash
   python3 <repo_root>/scripts/detect_faces.py --batch 256 --dataset $DATASET_NAME --resume
   ```

3. Head pose estimation 

   ```bash
   python3 <repo_root>/scripts/estimate_head_poses.py --batch 256 --dataset $DATASET_NAME
   ```

4. Faces tracking

   ~needs to be tested more -- not reliable to use in other places~

   ```bash
   python3 <repo_root>/scripts/track_faces.py --closest --batch 256 --dataset $DATASET_NAME
   ```

5. Eye landmarks estimation 

   In this step, eye landmarks are estimated using Mediapipe's PyTorch implementation of [face mesh](https://github.com/thepowerfuldeez/facemesh.pytorch) and [iris landmarks](https://github.com/cedriclmenard/irislandmarks.pytorch). It also computes the distance between the upper and lower eyelids using [Hausdorff_distance](https://en.wikipedia.org/wiki/Hausdorff_distance). Using the average positions between the upper and lower eyelids a curve is obtained, and along this curve the standard deviation of the color values is extracted.

   ```bash
   python3 <repo_root>/scripts/predict_eye_landmarks.py --batch 256 --dataset $DATASET_NAME --dim 2D
   ```

6. Signals generation

   for each video the following files are generated as frame-series signals:

   * eyelids_dists.pkl
   * face_not_found.pkl
   * stds.pkl: 3 signals for each color value (RGB)
   * yaw_angles.pkl
   * pitch_angles.pkl

   ```bash
   python3 <repo_root>/scripts/generate_signals.py --select_eye best --dataset $DATASET_NAME
   ```

### Training

1. Data preparation

   ```bash
   python3 <repo_root>/training/data_preparation.py \
   --output_folder <repo_root>/dataset/augmented_signals/versions \
   --suffix vtest 
   --overlap 10 \
   --face_found \
   --yaw_range 45 \
   --pitch_range 30 \
   --equal \
   --dataset $DATASET_NAME \
   --generate_plots
   ```

2. Train

   ```bash
   python3 <repo_root>/training/train_blinkdetector.py \
   --annotation_file <repo_root>/dataset/augmented_signals/versions/$DATASET_NAME/annotations-vtest.json \
   --dataset_path <repo_root>/dataset/augmented_signals/versions/$DATASET_NAME/vtest/training \
   --prefix archtest \
   --channels 1C  \
   --epoch 50 \
   --batch 4 \
   --normalized \
   --generate_fnfp_plots
   ```

   this will store a model in the folder `<repo_root>/`checkpoints with the name `<prefix>-<normalized>-<channels>-<epoch>.pth`. An example in evaluation section will be presented.

### Evaluation

1. Data Preparation

   ```bash
   # data preparation
   python3 <repo_root>/training/data_preparation.py \
   --output_folder <repo_root>/dataset/augmented_signals/versions \
   --suffix vtest \
   --overlap 15 \
   --face_found \
   --eval \
   --dataset $DATASET_NAME \
   --generate_plots
   ```

2. Evaluate

   ```bash
   DATASET_NAME=<FILL HERE>
   MODELS_FOLDER=checkpoints && \
   python3 <repo_root>/training/evaluate_softmax.py \
   --model ./$MODELS_FOLDER/archtest-False-1C-50.pth \
   --annotation_file <repo_root>/dataset/augmented_signals/versions/$DATASET_NAME/annotations-vtest.json \
   --dataset_path <repo_root>/dataset/augmented_signals/versions/$DATASET_NAME/vtest/training \
   --dataset $DATASET_NAME \
   --generate_fnfp_plots
   ```

## SVM with RT-BENE
[link](./examples/rtbene/README.md)