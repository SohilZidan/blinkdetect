#!/usr/bin/env python3
# coding: utf-8


import os
import argparse
import pandas as pd
import tqdm
import cv2

from blinkdetect.headpose import HeadPoseEstimator


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

    # headpose estimator
    pose_estimator = HeadPoseEstimator()

    # initialize
    yaws = []
    picthes = []
    rolls = []
    nons = 0

    # iterate over all items
    file_paths = temporal_blinking["file_path"].values
    for idx, file_path in enumerate(tqdm.tqdm(file_paths, total=len(file_paths))):
        # retrieve face info
        facial_area = temporal_blinking.iloc[idx]["facial_area"]

        if len(facial_area) == 0:
            nons = nons + 1
            yaw = 0
            pitch = 0
            roll = 0
        else:
            img = cv2.imread(file_path, cv2.IMREAD_COLOR)
            boxes = [facial_area]
            yaw, pitch, roll = pose_estimator.estimate_headpose(img, boxes)

        # append the new data
        yaws.append(yaw)
        picthes.append(pitch)
        rolls.append(roll)

    # store pose
    temporal_blinking["yaw"] = yaws
    temporal_blinking["pitch"] = picthes
    temporal_blinking["roll"] = rolls

    print(temporal_blinking.head())
    print(temporal_blinking.columns)
    print(f"{nons} faces are not detected")

    # create the parent directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    temporal_blinking.to_hdf(args.output, "temporal_blinking")
