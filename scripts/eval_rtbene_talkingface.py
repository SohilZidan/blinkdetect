from pandas.core import frame
import tqdm
import json
import os
import argparse
import pickle
import glob
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def read_annotations_tag(input_file: str):
    """read annotations by blinkmatters.com
    """
    name, ext = os.path.splitext(input_file)
    assert ext==".tag", "file extension is not .tag"
    # define variables 
    blink_start = 1
    blink_end = 1
    blink_info = (0,0)
    closeness_list = {}
    

    # Using readlines() 
    file1 = open(input_file, "r", encoding="utf-8") 
    Lines = file1.readlines() 

    # find "#start" line 
    start_line = 1
    for line in Lines: 
        clean_line=line.strip()
        if clean_line=="#start":
            break
        start_line += 1

    # convert tag file to readable format and build "closeness_list" and "blink_list"
    for index in range(len(Lines[start_line : -1])): # -1 since last line will be"#end"
        
        # read previous annotation and current annotation 
        prev_annotation=Lines[start_line+index-1].split(':')
        current_annotation=Lines[start_line+index].split(':')
        
        # if previous annotation is not "#start" line and not "blink" and current annotation is a "blink"
        if prev_annotation[0] != "#start\n" and prev_annotation[1] == "-1" and int(current_annotation[1]) > 0:
            # it means a new blink starts so save frame id as starting frame of the blink
            blink_start = int(current_annotation[0])
        
        # if previous annotation is not "#start" line and is a "blink" and current annotation is not a "blink"
        if prev_annotation[0] != "#start\n" and int(prev_annotation[1]) > 0 and current_annotation[1] == "-1":
            # it means a new blink ends so save (frame id - 1) as ending frame of the blink
            blink_end = int(current_annotation[0]) - 1
        
        # if current annotation consist fully closed eyes, append it also to "closeness_list" 
        if (current_annotation[3] == "C" and current_annotation[5] == "C"):
            closeness_list[f"{int(current_annotation[0]):06d}"] = 1.
        else:
            closeness_list[f"{int(current_annotation[0]):06d}"] = 0.
    
    file1.close()
    return closeness_list

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eyecutouts",
        required=True
    )
    parser.add_argument(
        "--annotations",
        required=True
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse()

    eyecutouts_folder = args.eyecutouts

    meta_file = os.path.join(eyecutouts_folder, "meta.pkl")
    #
    # video paths
    #
    preds_paths = []
    frames_paths = []
    for root, dirs, files in os.walk(eyecutouts_folder):
        for _file in files:
            name, ext = os.path.splitext(_file)
            if ext in [".pkl"]:
                if name == "frames":
                    frames_paths.append(os.path.join(root, _file))
                if name == os.path.basename(eyecutouts_folder):
                    preds_paths.append(os.path.join(root, _file))

    print(preds_paths)
    results = {}
    vid_progress = tqdm.tqdm(
        list(zip(preds_paths, frames_paths)), total=len(preds_paths),
        desc="participants")
    for preds_path, frames_path in vid_progress:
        video_name = os.path.dirname(preds_path)
        video_name = os.path.relpath(video_name, eyecutouts_folder)
        vid_progress.set_postfix(vid=video_name)
        #
        # read data
        #
        with open(preds_path, "rb") as f:
            closed_eyes = pickle.load(f)
        with open(frames_path, "rb") as f:
            frames = pickle.load(f)
        
        annotaions_file_tag = args.annotations
        closeness_list_dict = read_annotations_tag(annotaions_file_tag)

        frames = sorted(frames)
        closeness_list = []
        closed_eyes_preds = []
        for _idx, _frame in enumerate(frames):
            frames_seg = _frame.split("_")
            if len(frames_seg) == 3:
                _frame = frames_seg[1]
            elif len(frames_seg) == 1:
                _frame = frames_seg[0]
            # annotations start from 0, frames start from 1
            actual_frame_num = f"{(int(_frame)-1):06d}"
            if actual_frame_num in closeness_list_dict.keys():
                closeness_list.append(
                    closeness_list_dict[actual_frame_num])
                closed_eyes_preds.append(closed_eyes[_idx])
        #
        closed_eyes = closed_eyes_preds

        
        frames_seg = frames[0].split("_")
        if len(frames_seg) == 3:
            shift_amount = frames_seg[1]
        elif len(frames_seg) == 1:
            shift_amount = frames_seg[0]
        shift_amount = int(shift_amount) - 1

        results[video_name] = confusion_matrix(closeness_list, closed_eyes)
        print(results[video_name])
        print(classification_report(closeness_list, closed_eyes))

    with open(meta_file, "wb") as f:
        pickle.dump(results, f)
