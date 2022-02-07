#!/usr/bin/env python3
# coding: utf-8


import os
import glob
import argparse
import pandas as pd
import shutil
import tqdm
import cv2


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
    parser.add_argument(
        "--samples",
        required=True, type=int,
        help="number of samples to visualize"
        )
    return parser.parse_args()


BLINK_COLOR = (0, 0, 255)
NO_BLINK_COLOR = (0, 255, 0)


def overlay_gt_over_img(img, p, border_size=5):
    img_copy = img.copy()
    h, w = img_copy.shape[:2]
    if p > 0.5:
        cv2.rectangle(img_copy, (0, 0), (w, h), BLINK_COLOR, border_size)
    else:
        cv2.rectangle(img_copy, (0, 0), (w, h), NO_BLINK_COLOR, border_size)
    return img_copy


if __name__ == "__main__":
    args = parse()

    temporal_blinking = pd.read_hdf(args.annotations, "temporal_blinking")
    temporal_blinking = temporal_blinking.sort_index()

    samples_indexes = None
    chosen_samples_count_nonzero = 0
    while True:
        tmp_idx = temporal_blinking.sample(args.samples).index
        samples_indexes = tmp_idx.unique()
        for idx in samples_indexes:
            if idx[2] != 0:
                chosen_samples_count_nonzero = chosen_samples_count_nonzero + 1
        if samples_indexes.shape[0] == args.samples and (chosen_samples_count_nonzero > args.samples/2):
            break

    for sample_idx in samples_indexes:

        sample_df = temporal_blinking.loc[sample_idx]

        sub = sample_idx[0]
        label = sample_idx[2]
        interval = sample_idx[3]

        output_dir = os.path.join(args.output, "-".join([sub, str(label), interval]))
        os.makedirs(output_dir, exist_ok=True)

        file_paths = sample_df["file_path"].values
        scores = sample_df["score"].values

        for idx, file_path in enumerate(file_paths):
            score = scores[idx]
            frame_name = os.path.basename(file_path)
            frame_dst_path = os.path.join(output_dir, frame_name)
            # read image
            raw_img = cv2.imread(file_path, cv2.IMREAD_COLOR)
            # overlay
            overlayed_img = overlay_gt_over_img(raw_img, score)
            # save image
            cv2.imwrite(frame_dst_path, overlayed_img)

    # subject_level = temporal_blinking.index.unique(0).tolist()
    # for sub in tqdm.tqdm(subject_level, total=len(subject_level), desc="subject"):
    #     label_level = temporal_blinking.loc[sub].index.unique(0).tolist()
    #     # print(label_level)
    #     for label in tqdm.tqdm(label_level, total=len(label_level), desc="label", leave=False):
    #         time_series_level = temporal_blinking.loc[sub, label].index.unique(0).tolist()
    #         # print(time_series_level)
    #         for interval in tqdm.tqdm(time_series_level, total=len(time_series_level), desc="example", leave=False):

    #             output_dir = os.path.join(args.output, sub, str(label), interval)
    #             # ensure the dir exists
    #             os.makedirs(output_dir, exist_ok=True)

    #             tmp_view = temporal_blinking.loc[sub,label, interval]
    #             file_paths = tmp_view["file_path"].values
    #             scores = tmp_view["score"].values

    #             for idx, file_path in enumerate(file_paths):
    #                 score = scores[idx]

    #                 frame_name = os.path.basename(file_path)
    #                 frame_dst_path = os.path.join(output_dir, frame_name)
    #                 shutil.copyfile(file_path, frame_dst_path)
