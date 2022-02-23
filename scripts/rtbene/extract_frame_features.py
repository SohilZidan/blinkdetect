#!/usr/bin/env python3
# coding: utf-8


import os
import argparse
import pandas as pd
import tqdm
import cv2

from blinkdetect.eyelandmarks import IrisHandler

import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.6)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations",
        required=True,
        help="annotations file"
        )
    parser.add_argument(
        "--output",
        required=True,
        help="output folder"
        )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse()

    # read annotations
    temporal_blinking = pd.read_hdf(args.annotations, "temporal_blinking")
    temporal_blinking = temporal_blinking.sort_index()

    # iris handler
    iris_marker = IrisHandler()

    # initialize
    eyelids_dists = {"left": [], "right": []}
    iris_diameters = {"left": [], "right": []}
    pupil_dists = {"left": [], "right": []}
    nons = 0

    # iterate over all items
    file_paths = temporal_blinking["file_path"].values
    for idx, file_path in enumerate(tqdm.tqdm(file_paths, total=len(file_paths))):
        # retrieve face info
        facial_area = temporal_blinking.iloc[idx]["facial_area"]
        left_eye = temporal_blinking.iloc[idx]["left_eye"]
        right_eye = temporal_blinking.iloc[idx]["right_eye"]
        face_landmarks = {
            "facial_area" : facial_area,
            "landmarks": {
                "left_eye": left_eye,
                "right_eye": right_eye
                }
            }

        img = cv2.imread(file_path, cv2.IMREAD_COLOR)

        if len(facial_area) == 0:
            nons = nons + 1
            eyelidsDistances = {"left": -1, "right": -1}
            irisesDiameters = {"left": -1, "right": -1}
            pupil2corners = {"left": -1, "right": -1}
        else:
            eyelidsDistances, irisesDiameters, pupil2corners = iris_marker.extract_features(img, True, face_landmarks)

        # append the new data
        for eye_idx in eyelidsDistances.keys():
            eyelids_dists[eye_idx].append(eyelidsDistances[eye_idx])
            iris_diameters[eye_idx].append(irisesDiameters[eye_idx])
            pupil_dists[eye_idx].append(pupil2corners[eye_idx])





    # args = parse()

    # read annotations
    # temporal_blinking = pd.read_hdf(args.annotations, "temporal_blinking")
    # temporal_blinking = temporal_blinking.sort_index()

    # # add the new columns
    # for eye_idx in ["left", "right"]:
    #     temporal_blinking[f"{eye_idx}_eyelids_dist"] = np.nan
    #     temporal_blinking[f"{eye_idx}_iris_diameter"] = np.nan
    #     temporal_blinking[f"{eye_idx}_pupil2corner"] = np.nan
    # temporal_blinking["facial_area"] = np.nan
    # temporal_blinking["left_eye"] = np.nan
    # temporal_blinking["right_eye"] = np.nan

    # print(temporal_blinking.head())

    # initialize
    # iris_marker = IrisHandler()
    # eyelids_dists = {"left": [], "right": []}
    # iris_diameters = {"left": [], "right": []}
    # pupil_dists = {"left": [], "right": []}
    # faces = []
    # left_eyes = []
    # right_eyes = []
    # nons = 0

    # # iterate over each subject, label, interval
    # subject_level = temporal_blinking.index.unique(0).tolist()
    # for sub in tqdm.tqdm(subject_level, total=len(subject_level), desc="subject"):

    #     label_level = temporal_blinking.loc[sub].index.unique(0).tolist()

    #     for label in tqdm.tqdm(label_level, total=len(label_level), desc="label", leave=False):

    #         time_series_level = temporal_blinking.loc[sub, label].index.unique(0).tolist()

    #         for interval in tqdm.tqdm(time_series_level, total=len(time_series_level), desc="example", leave=False):

    #             tmp_view = temporal_blinking.loc[(sub, label, interval)]
    #             file_paths = tmp_view["file_path"].values
    #             scores = tmp_view["score"].values
    #             facial_areas = tmp_view["facial_area"].values
    #             left_eyes = tmp_view["left_eye"].values
    #             right_eyes = tmp_view["right_eye"].values

    #             # extract features here
    #             for idx, file_path in enumerate(file_paths):

    #                 # read image
    #                 img = cv2.imread(file_path, cv2.IMREAD_COLOR)

    #                 # retrieve face
    #                 face_landmarks = {
    #                     "facial_area" : facial_areas[idx],
    #                     "landmarks": {
    #                         "left_eye": left_eyes[idx],
    #                         "right_eye": right_eyes[idx]
    #                         }
    #                     }

    #                 # analyze
    #                 # each element of the tuple is a dict of keys: [left, right]
    #                 if not isinstance(facial_areas[idx], list):# np.isnan(facial_areas[idx]):
    #                     nons = nons + 1
    #                     eyelidsDistances = {"left": np.nan, "right": np.nan}
    #                     irisesDiameters = {"left": np.nan, "right": np.nan}
    #                     pupil2corners = {"left": np.nan, "right": np.nan}
    #                     # faces.append(np.nan)
    #                     # left_eyes.append(np.nan)
    #                     # right_eyes.append(np.nan)
    #                 else:
    #                     try:
    #                         eyelidsDistances, irisesDiameters, pupil2corners = iris_marker.extract_features(img, True, face_landmarks)
    #                     except Exception as e:
    #                         print(e)
    #                         exit()
    #                     # faces.append(face_landmarks["facial_area"])
    #                     # left_eyes.append(face_landmarks["landmarks"]["left_eye"])
    #                     # right_eyes.append(face_landmarks["landmarks"]["right_eye"])
    
    #                 # append the new data
    #                 for eye_idx in ["left", "right"]:
    #                     eyelids_dists[eye_idx].append(eyelidsDistances[eye_idx])
    #                     iris_diameters[eye_idx].append(irisesDiameters[eye_idx])
    #                     pupil_dists[eye_idx].append(pupil2corners[eye_idx])

                # print(temporal_blinking.loc[(sub, label, interval)].iloc[idx])

    for eye_key in eyelids_dists.keys():
        temporal_blinking[f"{eye_key}_eyelids_dist"] = eyelids_dists[eye_key]
        # temporal_blinking["right_eyelids_dist"] = eyelids_dists["right"]
        temporal_blinking[f"{eye_key}_diameter"] = iris_diameters[eye_key]
        # temporal_blinking["right_iris_diameter"] = iris_diameters["right"]
        temporal_blinking[f"{eye_key}_pupil2corner"] = pupil_dists[eye_key]
        # temporal_blinking["right_pupil2corner"] = pupil_dists["right"]
    # faceinfo
    # temporal_blinking["facial_are"] = faces
    # temporal_blinking["left_eye"] = left_eyes
    # temporal_blinking["right_eye"] = right_eyes

    print(temporal_blinking.head())
    print(temporal_blinking.columns)
    print(f"{nons} faces are not detected")

    # create the parent directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    temporal_blinking.to_hdf(args.output, "temporal_blinking")
