from pandas.core import frame
import tqdm
import json
import os
import argparse
import pickle
import glob
import numpy as np
from argusutil.evaluation import IoUStrictMatchConfMat, FMeasure
from blinkdetect.argus_utils import get_intervals, get_intervals_between
from blinkdetect.argus_utils import get_blinking_annotation
from blinkdetect.common import read_annotations_tag
from sklearn.metrics import classification_report, confusion_matrix


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', required=True,
        choices=[
            "BlinkingValidationSetVideos", "eyeblink8",
            "talkingFace", "zju", "RN"]
            )
    parser.add_argument(
        "--yaw_range", type=int, metavar='N',  nargs=2, default=[180])
    parser.add_argument(
        "--pitch_range", type=int, metavar='M',  nargs=2, default=[180])
    parser.add_argument(
        "--face_found", action="store_true")
    args = parser.parse_args()
    return args


def evaluate(prd_blinking_annotation, gt_blinking_annotation):
    fmeasure = FMeasure(IoUStrictMatchConfMat(0.3))
    fmeasure.evaluate(
        {"Blinking": gt_blinking_annotation},
        {"Blinking": prd_blinking_annotation})
    return fmeasure.to_dict()


if __name__ == "__main__":
    args = parse()
    print(args)
    dataset_root = os.path.join(os.path.dirname(__file__), "..", "dataset")
    eyecutouts_folder = os.path.join(dataset_root, "eye-cutouts", args.dataset)
    meta_file = os.path.join(eyecutouts_folder, "meta.json")
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
                else:
                    preds_paths.append(os.path.join(root, _file))

    results = {}
    vid_progress = tqdm.tqdm(
        list(zip(preds_paths, frames_paths)), total=len(preds_paths),
        desc="participants")

    confusion_matrix_all = None

    for preds_path, frames_path in vid_progress:
        video_name = os.path.dirname(preds_path)
        video_name = os.path.relpath(video_name, eyecutouts_folder)
        vid_progress.set_postfix(vid=video_name)
        #
        # read data
        #
        signals_path = os.path.join(
            dataset_root, "tracked_faces", args.dataset, video_name, "signals")
        faces_not_found_path = os.path.join(signals_path, "face_not_found.pkl")
        yaw_path = os.path.join(signals_path, "yaw_angles.pkl")
        pitch_path = os.path.join(signals_path, "pitch_angles.pkl")
        with open(faces_not_found_path, "rb") as f:
            faces_not_found = pickle.load(f)
        with open(yaw_path, "rb") as f:
            yaw_angles = pickle.load(f)
        with open(pitch_path, "rb") as f:
            pitch_angles = pickle.load(f)
        with open(preds_path, "rb") as f:
            closed_eyes = pickle.load(f)
        with open(frames_path, "rb") as f:
            frames = pickle.load(f)
        #
        if args.dataset == "BlinkingValidationSetVideos":
            blinking_anns = get_blinking_annotation(video_name)
        else:
            video_path = os.path.join(dataset_root, args.dataset, video_name)
            annotation_paths = glob.glob(video_path + "/*.tag")
            if len(annotation_paths) != 1:
                exit()
            annotaions_file_tag = annotation_paths[0]
            closeness_list_dict, blinking_anns = read_annotations_tag(annotaions_file_tag)
        #
        frames = sorted(frames)
        # for _frame_d in range(int(frames[0]), int(frames[-1])):
        closeness_list = []
        closed_eyes_preds = []
        for _idx, _frame in enumerate(frames):
            if _frame not in frames:
                closed_eyes = (
                    closed_eyes[:_idx]
                    + [[False]]
                    + closed_eyes[_idx:]
                    )
            if args.dataset != "BlinkingValidationSetVideos":
                # because frames files names start from 1
                # annotations start from 0
                actual_frame_num = int(_frame) - 1
                
                closeness_list.append(
                    closeness_list_dict[f"{actual_frame_num:06d}"])
                closed_eyes_preds.append(closed_eyes[_idx])
        if args.dataset != "BlinkingValidationSetVideos":
            closed_eyes = closed_eyes_preds
        # for _idx, _frame in enumerate(frames):
            # if _frame in closeness_list_dict:

        closed_eyes = [i for sublist in closed_eyes for i in sublist]
        closed_eyes = (1 * np.array(closed_eyes)).tolist()
        #
        face_found_anns = get_intervals(faces_not_found, val=0)
        yaw_preds = get_intervals_between(yaw_angles, val=args.yaw_range)
        pitch_preds = get_intervals_between(pitch_angles, val=args.pitch_range)
        #
        blink_preds = get_intervals(closed_eyes, val=1)
        blink_preds = blink_preds.shift(int(frames[0])-1)
        blink_preds = blink_preds.intersect(face_found_anns)
        blink_preds = blink_preds.intersect(yaw_preds)
        blink_preds = blink_preds.intersect(pitch_preds)
        #
        blinking_anns = blinking_anns.intersect(face_found_anns)
        blinking_anns = blinking_anns.intersect(yaw_preds)
        blinking_anns = blinking_anns.intersect(pitch_preds)

        results[video_name] = {
            "metric": evaluate(blink_preds, blinking_anns)
            }
        print(list(zip(blink_preds, blinking_anns)))
        print(results[video_name]['metric'])
        if args.dataset != "BlinkingValidationSetVideos":
            if confusion_matrix_all is None:
                confusion_matrix_all = confusion_matrix(closeness_list, closed_eyes)
            else:
                confusion_matrix_all += confusion_matrix(closeness_list, closed_eyes)
            print(confusion_matrix(closeness_list, closed_eyes))
            print(classification_report(closeness_list, closed_eyes))

    if confusion_matrix_all is not None:
        print(confusion_matrix_all)
    with open(meta_file, "w") as f:
        json.dump(results, f)
