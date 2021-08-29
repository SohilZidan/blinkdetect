### requirements:
* python 3.8

### components and pipeline
1. face detection: using retina-face python package 0.0.4
2. face head pose estimation: 3DDFA_V2
3. face tracking: !! KLT and deepface
4. eye landmarks estimation: face mesh estimation: Mediapipe -- file eye_landmarks_predict append new information to the output of the face tracker
5. signals generation


### TODO
* debugging code -- logging
* configs


### Building
* clone this repository: `git clone --recursive <project url>`
* cd <project url>
* sh ./build.sh

### Steps
1. generate frames of the videos
   * place videos inside <repo_root>/dataset/BlinkingValidationSetVideos
   * run generate_frames_v2.sh
   * the output structure of the videos is as follows:
      ```
      ├── BlinkingValidationSetVideos
      │   ├── video_1.ext
      |   ├── video_1
      │   │   ├── frames
      │   │   |   ├── 000000.png
      │   │   |   ├── 000001.png
      │   │   |   ├── ...
      │   ├── video_2.ext
      │   ├── ...
      │   └── video_n.ext
      ├── ...
      └── ...
      ```
2. 