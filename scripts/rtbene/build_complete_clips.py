#!/usr/bin/env python3
# coding: utf-8


import os
import argparse
import shutil
import pandas as pd
import tqdm
import cv2
# from blinkdetect.common import read_annotations_tag, read_bbox_rush
# from blinkdetect.image.misc import cut_region



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

    # iterate over subjects -> complete intervals
    subject_level = temporal_blinking.index.unique(0).tolist()
    for sub in tqdm.tqdm(subject_level, total=len(subject_level), desc="subject"):
        complete_interval_level = temporal_blinking.loc[sub].index.unique(0).tolist()
        for interval in tqdm.tqdm(complete_interval_level, total=len(complete_interval_level), desc="example", leave=False):
            interval_view = temporal_blinking.loc[(sub, interval)]

            # get paths
            frames_paths = interval_view["file_path"].tolist()

            # create dir
            complete_interval_dir = os.path.join(args.output, sub, interval)
            if os.path.exists(complete_interval_dir):
                shutil.rmtree(complete_interval_dir)
            os.makedirs(complete_interval_dir)

            # save path
            for src_path in frames_paths:
                file_name = os.path.basename(src_path)
                dest_path = os.path.join(complete_interval_dir, file_name)
                shutil.copyfile(src_path, dest_path)
