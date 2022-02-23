#!/usr/bin/env python3
# coding: utf-8


import os
import argparse
import tqdm
import pandas as pd

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
        "--threshold", type=int, default=30,
        help="minimum number of consecutive frames, and window size"
        )
    parser.add_argument(
        "--step",
        required=True, type=int,
        help="number of frames to skip when shifting the window"
        )

    return parser.parse_args()


def compute_label(_group: pd.DataFrame) -> int:
    # annotate:
    # open: 0
    # closed: 1
    # time-series annotations:
    # 0...-> 1...: closing 2
    # 1...-> 0...: opening 3
    # 0...-> 1...-> 0...: blink 4
    # 1...-> 0...-> 1...: other 5
    first = True
    state_order = []
    prev_state = None
    for row in _group.iterrows():
        if first:
            prev_state = (row[1]["score"] > 0)
            state_order.append(prev_state)
            first = False
        else:
            current_state = (row[1]["score"] > 0)
            if current_state != prev_state:
                state_order.append(current_state)
            prev_state = current_state

    # check state_order
    label = -1
    if len(state_order) == 3:
        # blink
        if state_order == [False, True, False]:
            label = 4
        # other
        if state_order == [True, False, True]:
            label = 5
    elif len(state_order) == 2:
        # closing
        if state_order == [False, True]:
            label = 2
        # opening
        if state_order == [True, False]:
            label = 3
    elif len(state_order) == 1:
        # closed, open
        label = 1 if state_order[0] else 0

    return label


def get_valid_intervals(group: pd.DataFrame):
    # initialize
    frames_count = group.shape[0]
    sub_groups = []
    prev_found = False
    start = 0

    for idx in range(0,frames_count):
        current_found = len(group.iloc[idx]["facial_area"]) != 0
        # if not current_found: print(group.iloc[idx]["facial_area"], group.iloc[idx]["file_path"])
        if current_found:
            if not prev_found:
                start = idx
        else:
            if prev_found:
                if not group.iloc[start:idx].empty:
                    sub_groups.append(group.iloc[start:idx])
                    # if group.iloc[start:idx].shape[0] == 30:
                    #     print(group.iloc[start:idx])
                    #     exit()
        prev_found = current_found
    
    if (start < frames_count-1):
        sub_groups.append(group.iloc[start:])

    return sub_groups

extract_frame = {
        "file_name": lambda x: x.split('_')[1]
        }


if __name__ == "__main__":

    args = parse()

    # read annotations
    temporal_blinking = pd.read_hdf(args.annotations, "temporal_blinking")
    temporal_blinking = temporal_blinking.sort_index()
    # print(temporal_blinking.shape)
    # print(temporal_blinking.head())
    # print(temporal_blinking['facial_area'])
    # exit()

    # initialize
    names = ["subject", "frame_range", "label", "sub_frame_range"]
    tmp_index = []
    data = []

    subject_level = temporal_blinking.index.unique(0).tolist()
    for sub in tqdm.tqdm(subject_level, total=len(subject_level), desc="subject"):
        time_series_level = temporal_blinking.loc[sub].index.unique(0).tolist()
        for interval in tqdm.tqdm(time_series_level, total=len(time_series_level), desc="example", leave=False):
            interval_view = temporal_blinking.loc[(sub, interval)]


            frames_count = interval_view.shape[0]
            for start in range(0, frames_count, args.step):
                end = start + args.threshold
                # take #threshold rows
                sub_group = interval_view.iloc[start:end]

                # get_valid_intervals: intervals where faces are detected
                valid_groups = get_valid_intervals(sub_group)

                for valid_group in valid_groups:
                    # when frames_count is not threshold divisible
                    if valid_group.shape[0] == args.threshold:
                        # process data
                        sub_data = valid_group.values.tolist()
                        data.extend(sub_data)
                        # process index
                        label = compute_label(valid_group)
                        frame_range = "-".join([str(valid_group["frame"].values[0]), str(valid_group["frame"].values[-1])])
                        sub_index = list(zip(
                                [sub]*args.threshold, [interval]*args.threshold,
                                [label]*args.threshold, [frame_range]*args.threshold,
                            ))
                        tmp_index.extend(sub_index)

    # build multiindex dataframe
    index = pd.MultiIndex.from_tuples(tmp_index, names=names)
    temporal_blinking = pd.DataFrame(data, index=index, columns=temporal_blinking.columns)
    print(temporal_blinking.head())
    print(temporal_blinking.shape[0])
    # for i in range(-1,6):
    #     print(f"{i}:", temporal_blinking.loc[(slice(None), slice(None), i, ), :].index.unique().shape[0])

    # prepare output path
    parent_dir = os.path.dirname(args.output)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    # save into hdf5
    temporal_blinking.to_hdf(args.output, "temporal_blinking")
