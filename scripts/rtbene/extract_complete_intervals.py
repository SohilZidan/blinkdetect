#!/usr/bin/env python3
# coding: utf-8

import os
import glob
import argparse
from typing import List
import numpy as np
import pandas as pd
from blinkdetect.common import read_annotations_tag, read_bbox_rush
from blinkdetect.argus_utils import get_closed_eye_annotations


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subject",
        required=True,
        help="subject identifier"
        )
    parser.add_argument(
        "--blink_annotations",
        required=True,
        help="eye patches dir"
        )
    parser.add_argument(
        "--faces",
        required=True,
        help="faces dir"
        )
    parser.add_argument(
        "--output",
        required=True,
        help="output folder"
        )
    parser.add_argument(
        "--threshold", type=int, default=30,
        help="minimum number of consecutive frames, and window size"
        )
    parser.add_argument(
        "--dataset", type=str, default="rtbene",
        choices=["rush", "rtbene", "talkingFace", "rn"],
        help="minimum number of consecutive frames, and window size"
        )
    return parser.parse_args()


def consecutive_frames(blink_data: pd.DataFrame, face_paths: List, threshold: int) -> List[pd.DataFrame]:
    frames_groups = []
    frames_group = []
    prev_idx = -1
    # frames names
    for face_path in face_paths:
        file_name = os.path.basename(face_path)
        split_parts = file_name.split("_")
        if len(split_parts) == 1:
            # anns are 0-indexed for talkingFace
            # eliminate -1 for rush
            # frame_idx = int(file_name.split(".")[0]) - 1
            frame_idx = int(file_name.split(".")[0])
        else:
            frame_idx = int(file_name.split("_")[1])

        if (frame_idx - prev_idx) > 1:
            # add to the list of groups
            if len(frames_group) > threshold:
                frames_groups.append(frames_group)
            # flush
            frames_group = []

        if frame_idx in blink_data.index and (blink_data.loc[frame_idx].item() < 0.5 or blink_data.loc[frame_idx].item() > 0.5):
            _item = {"frame":frame_idx, "file_name": file_name, "file_path": face_path, "score": blink_data.loc[frame_idx].item()}
            frames_group.append(_item)
            prev_idx = frame_idx

    if len(frames_group) >= threshold:
        frames_groups.append(frames_group)

    frames_groups = [pd.DataFrame(_group).set_index("frame") for _group in frames_groups]

    return frames_groups


extract_frame = {
        "file_name": lambda x: x.split('_')[1]
        }


if __name__ == "__main__":

    args = parse()

    blinks_anns_file = args.blink_annotations
    assert os.path.exists(blinks_anns_file), f"annotations file {blinks_anns_file} does not exist"

    # read blink annotations -- csv
    if args.dataset in ["talkingFace", "rn"]:
        clossnes, _ = read_annotations_tag(blinks_anns_file)
        df = pd.DataFrame(
            {
                "file_name": pd.Series(clossnes.keys()).astype("int32"), 
                "score": pd.Series(clossnes.keys()).astype("int32")
                })
        blink_data = df.set_index("file_name")
    elif args.dataset == "rtbene":
        blink_data = pd.read_csv(blinks_anns_file, header=None, names=["file_name", "score"], index_col=0, converters=extract_frame)
    elif args.dataset == "rush":
        face_bboxes = read_bbox_rush(args.blink_annotations)
        closed_eyes_idxes = get_closed_eye_annotations(args.subject)
        frames_ids = sorted(face_bboxes.keys()) # filename with ext
        #
        frames = []
        scores = []
        for frame_id in frames_ids:
            frame_name, _ = os.path.splitext(frame_id)
            frame_name = int(frame_name)
            frames.append(frame_name)
            if frame_name in closed_eyes_idxes:
                scores.append(1)
            else:
                scores.append(0)
        #
        df = pd.DataFrame(
            {
                "file_name": pd.Series(frames, dtype=np.dtype("int32")), 
                "score": pd.Series(scores, dtype=np.dtype("int32"))
                })
        blink_data = df.set_index("file_name")

    # retrieve face paths
    faces_dir = os.path.normpath(args.faces)
    face_paths = sorted(glob.glob(f"{faces_dir}/*"))
    # print(face_paths)

    # extract consecutive frames
    # output is a list of pandas dataframes, where each dataframe
    # contains consecutive annotated frames
    frames_groups = consecutive_frames(blink_data, face_paths, args.threshold)
    print(f"{len(frames_groups)} groups of consecutive frames")
    total_frames = sum([_group.shape[0] for _group in frames_groups])
    print(f"with a total of {total_frames} frames")

    # multiindex dataframe
    names = ["subject", "frame_range"]
    tmp_index = []
    data = []

    # retireve label
    for sub_group in frames_groups:
        frames_count = sub_group.shape[0]

        # only take parts with >= threshold length
        if sub_group.shape[0] < args.threshold: continue

        frame_range = "-".join([str(sub_group.index[0]), str(sub_group.index[-1])])
        sub_data = list(zip(sub_group.index.values, sub_group["file_path"].values, sub_group['score'].values))
        data.extend(sub_data)
        sub_index = list(zip([args.subject]*sub_group.shape[0], [frame_range]*sub_group.shape[0],))
        tmp_index.extend(sub_index)

    # build multiindex dataframe
    index = pd.MultiIndex.from_tuples(tmp_index, names=names)
    temporal_blinking = pd.DataFrame(data, index=index, columns=["frame", "file_path", "score"])
    print(temporal_blinking.head())

    # prepare output path
    parent_dir = os.path.dirname(args.output)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    # save into hdf5
    temporal_blinking.to_hdf(args.output, "temporal_blinking", append=True, min_itemsize={"subject": 2, "frame_range": 20, "file_path": 70})
