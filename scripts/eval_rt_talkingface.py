from pandas.core import frame
import tqdm
import json
import os
import argparse
import pickle
import glob
import numpy as np
from blinkdetect.argus_utils import get_blinking_annotation
from blinkdetect.common import read_annotations_tag
from sklearn.metrics import classification_report, confusion_matrix


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        choices=[
            "BlinkingValidationSetVideos", "eyeblink8",
            "talkingFace", "zju", "RN"],
        default="talkingFace"
            )
    parser.add_argument(
        "--dataset_root",
        required=True,
        default=None
    )
    # parser.add_argument(
    #     "--yaw_range", type=int, metavar='N',  nargs=2, default=[180])
    # parser.add_argument(
    #     "--pitch_range", type=int, metavar='M',  nargs=2, default=[180])
    # parser.add_argument(
    #     "--face_found", action="store_true")
    args = parser.parse_args()
    return args





if __name__ == "__main__":
    args = parse()
    print(args)
    # CHANGE DIRName of talkingface
    if args.dataset_root is None:
        dataset_root = os.path.join(os.path.dirname(__file__), "..", "dataset")

    eyecutouts_folder = os.path.join(dataset_root, "eye-cutouts", args.dataset)
    # dataset_signals_path = os.path.join(dataset_root, "tracked_faces", args.dataset)
    frames_root = os.path.join(dataset_root, args.dataset)
    if args.dataset == "RN":
        eyecutouts_folder = os.path.join(eyecutouts_folder, "test")
        # dataset_signals_path = os.path.join(dataset_signals_path, "test")
        frames_root = os.path.join(frames_root, "test")

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

    all_anns = 0

    for preds_path, frames_path in vid_progress:
        video_name = os.path.dirname(preds_path)
        video_name = os.path.relpath(video_name, eyecutouts_folder)
        vid_progress.set_postfix(vid=video_name)
        #
        # read data
        #
        # signals_path = os.path.join(
        #     dataset_signals_path, video_name, "signals")
        # faces_not_found_path = os.path.join(signals_path, "face_not_found.pkl")
        # yaw_path = os.path.join(signals_path, "yaw_angles.pkl")
        # pitch_path = os.path.join(signals_path, "pitch_angles.pkl")
        # with open(faces_not_found_path, "rb") as f:
        #     faces_not_found = pickle.load(f)
        # with open(yaw_path, "rb") as f:
        #     yaw_angles = pickle.load(f)
        # with open(pitch_path, "rb") as f:
        #     pitch_angles = pickle.load(f)
        with open(preds_path, "rb") as f:
            closed_eyes = pickle.load(f)
        with open(frames_path, "rb") as f:
            frames = pickle.load(f)
        print(len(closed_eyes))
        #
        if args.dataset == "BlinkingValidationSetVideos":
            blinking_anns = get_blinking_annotation(video_name)
        else:
            video_path = os.path.join(frames_root, video_name)
            annotation_paths = glob.glob(video_path + "/*.tag")
            if len(annotation_paths) != 1:
                exit()
            annotaions_file_tag = annotation_paths[0]
            closeness_list_dict, blinking_anns = read_annotations_tag(annotaions_file_tag)
            all_anns += len(closeness_list_dict)
        #
        # closeness_list = [closeness_list_dict[_key] for _key in sorted(closeness_list_dict.keys())]

        frames = sorted(frames)
        closeness_list = []
        closed_eyes_preds = []
        for _idx, _frame in enumerate(frames):
            # if args.dataset != "BlinkingValidationSetVideos":
            #     # because frames files names start from 1
            #     # annotations start from 0
            frames_seg = _frame.split("_")
            if len(frames_seg) == 3:
                _frame = frames_seg[1]
            elif len(frames_seg) == 1:
                _frame = frames_seg[0]
            actual_frame_num = f"{(int(_frame)-1):06d}"

            if actual_frame_num in closeness_list_dict.keys():
                if _frame == "000001":
                    print("corresponding frames:", _frame, actual_frame_num)
                closeness_list.append(
                    closeness_list_dict[actual_frame_num])
                closed_eyes_preds.append(closed_eyes[_idx])
        with open("./res.txt", "w") as f:
            mylist = closed_eyes
            f.write('\n'.join('%s' % x for x in mylist))
        closed_eyes = closed_eyes_preds
        # if args.dataset != "BlinkingValidationSetVideos":
        #     closed_eyes = closed_eyes_preds
        # for _idx, _frame in enumerate(frames):
            # if _frame in closeness_list_dict:

        # closed_eyes_t = [] 
        # for sublist in closed_eyes:
        #     for i in sublist:
        #         if i: closed_eyes_t.append(1.)
        #         else: closed_eyes_t.append(0.)
        # closed_eyes = closed_eyes_t
        # closed_eyes = np.array(closed_eyes).tolist()
        #
        # face_found_anns = get_intervals(faces_not_found, val=0)
        # yaw_preds = get_intervals_between(yaw_angles, val=args.yaw_range)
        # pitch_preds = get_intervals_between(pitch_angles, val=args.pitch_range)
        # #
        # blink_preds = get_intervals(np.array(closed_eyes).tolist(), val=1)
        
        frames_seg = frames[0].split("_")
        if len(frames_seg) == 3:
            shift_amount = frames_seg[1]
        elif len(frames_seg) == 1:
            shift_amount = frames_seg[0]
        shift_amount = int(shift_amount) - 1

        # blink_preds = blink_preds.shift(shift_amount)
        # blink_preds = blink_preds.intersect(face_found_anns)
        # blink_preds = blink_preds.intersect(yaw_preds)
        # blink_preds = blink_preds.intersect(pitch_preds)
        # #
        # blinking_anns = blinking_anns.intersect(face_found_anns)
        # blinking_anns = blinking_anns.intersect(yaw_preds)
        # blinking_anns = blinking_anns.intersect(pitch_preds)

        # results[video_name] = {
        #     "metric": evaluate(blink_preds, blinking_anns)
        #     }
        # print(results[video_name]['metric'])
        if args.dataset != "BlinkingValidationSetVideos":
            if confusion_matrix_all is None:
                confusion_matrix_all = confusion_matrix(closeness_list, closed_eyes)
            else:
                confusion_matrix_all += confusion_matrix(closeness_list, closed_eyes)
            print(confusion_matrix(closeness_list, closed_eyes))
            print(classification_report(closeness_list, closed_eyes))

    if confusion_matrix_all is not None:
        print(confusion_matrix_all)

    # with open("./res.txt", "w") as f:
    #     mylist = list(zip(closeness_list, closed_eyes))
    #     f.write('\n'.join('%s %s' % x for x in mylist))
 
    print("annotated frames:", all_anns)
    with open(meta_file, "w") as f:
        json.dump(results, f)
