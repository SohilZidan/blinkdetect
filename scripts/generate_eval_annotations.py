#!/usr/bin/env python3

import os
import argparse
from argusutil.annotation.annotation import AnnotationOfIntervals, Interval, Unit

dataset_root = os.path.join(os.path.dirname(__file__), "..", "dataset")


def parser():
    _parser = argparse.ArgumentParser()
    _parser.add_argument('--dataset', required=True, choices=[
                         "BlinkingValidationSetVideos", "eyeblink8", "talkingFace", "zju", "RN"])
    return _parser.parse_args()


# https://www.kaggle.com/hakkoz/eye-blink-detection-1-simple-model?scriptVersionId=33335937&cellId=26
# read tag file and construct "closeness_list" and "blinks_list"
def read_annotations(input_file):
    # define variables
    blink_start = 1
    blink_end = 1
    blink_info = (0, 0)
    blink_list = []
    closeness_list = []

    # Using readlines()
    file1 = open(input_file)
    Lines = file1.readlines()

    # find "#start" line
    start_line = 1
    for line in Lines:
        clean_line = line.strip()
        if clean_line == "#start":
            break
        start_line += 1

    # convert tag file to readable format and build "closeness_list" and "blink_list"
    # -1 since last line will be"#end"
    for index in range(len(Lines[start_line: -1])):

        # read previous annotation and current annotation
        prev_annotation = Lines[start_line+index-1].split(':')
        current_annotation = Lines[start_line+index].split(':')

        # if previous annotation is not "#start" line and not "blink" and current annotation is a "blink"
        if prev_annotation[0] != "#start\n" and prev_annotation[1] == "-1" and int(current_annotation[1]) > 0:
            # it means a new blink starts so save frame id as starting frame of the blink
            blink_start = int(current_annotation[0])

        # if previous annotation is not "#start" line and is a "blink" and current annotation is not a "blink"
        if prev_annotation[0] != "#start\n" and int(prev_annotation[1]) > 0 and current_annotation[1] == "-1":
            # it means a new blink ends so save (frame id - 1) as ending frame of the blink
            blink_end = int(current_annotation[0]) - 1
            # and construct a "blink_info" tuple to append the "blink_list"
            blink_info = Interval(blink_start, blink_end)
            blink_list.append(blink_info)

        # if current annotation consist fully closed eyes, append it also to "closeness_list"
        if current_annotation[3] == "C" and current_annotation[5] == "C":
            closeness_list.append(1)

        else:
            closeness_list.append(0)

    file1.close()
    blinks_intervals = AnnotationOfIntervals(Unit.INDEX, blink_list)
    return closeness_list, blinks_intervals


if __name__ == "__main__":
    args = parser()
    dataset = os.path.join(dataset_root, args.dataset)
    #
    #
    # video paths
    anns_paths = []
    for root, dirs, files in os.walk(dataset):
        for dir in files:
            name, ext = os.path.splitext(dir)
            if ext in [".tag"]:
                anns_paths.append(os.path.join(root, dir))
    #
    #
    for anns_path in anns_paths:
        video_name = os.path.dirname(anns_path)
        # video_name = os.path.relpath(video_name, dataset)
        print(video_name, anns_path)
        closeness_list, blinks_intervals = read_annotations(anns_path)
        print(blinks_intervals)
