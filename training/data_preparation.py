#!/usr/bin/env python3
# coding: utf-8

import warnings
warnings.filterwarnings("ignore")

import os
import json
import shutil
from math import ceil, floor
import argparse
import pickle
import random
import tqdm
import numpy as np
import matplotlib.pyplot as plt

from argusutil.annotation.annotation import (
    AnnotationOfIntervals, Interval, Unit
    )
from blinkdetect.argus_utils import (
    get_intervals, get_intervals_between, get_blinking_annotation)
from blinkdetect.signal_1d import shift
from blinkdetect.preprocessing import (
    resample_noblink, upsample_blink, downsample_blink)
from blinkdetect.common import read_annotations_tag


def check_positive(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", default="")
    parser.add_argument('--dataset', required=True, choices=["BlinkingValidationSetVideos", "eyeblink8", "talkingFace", "zju", "RN"])
    parser.add_argument("--generate_plots", action="store_true")
    parser.add_argument("--yaw_range", type=int, metavar='N',  nargs=2, default=[180])
    parser.add_argument("--pitch_range", type=int, metavar='M',  nargs=2, default=[180])
    parser.add_argument("--face_found", action="store_true")
    parser.add_argument("--no_blink", type=check_positive, help="number of no blinks to be generated in between blinks")
    parser.add_argument("--overlap", type=int, choices=range(0,30) ,default=0, metavar="[0,30)")
    parser.add_argument("--suffix", default="-v0")
    parser.add_argument("--normalized", action="store_true")
    parser.add_argument("--equal", action="store_true", help="if set equal numbers of both classes ar generated")
    parser.add_argument("--zero", action="store_true", help="if set no noblink will be generated")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--ratio", type=float, default=0)
    args = parser.parse_args()
    return args

def main(cfgs):
    random.seed(192020)

    # OUTPUTs
    dataset_root = os.path.join(os.path.dirname(__file__), "..", "dataset")
    dataset = os.path.join(dataset_root, cfgs.dataset)
    if cfgs.output_folder == "":
        annotations_folder = os.path.join(dataset_root, "augmented_signals")
        cfgs.output_folder = annotations_folder
    else:
        annotations_folder = os.path.join(cfgs.output_folder, cfgs.dataset)
    os.makedirs(annotations_folder, exist_ok=True)

    _version_folder = os.path.join(annotations_folder, cfgs.suffix)
    meta_file = os.path.join(_version_folder, "meta.json")
    output_folder = os.path.join(_version_folder, "plots")

    if os.path.exists(_version_folder):
        shutil.rmtree(_version_folder)
    os.makedirs(_version_folder)

    if cfgs.generate_plots:
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder)

    # 
    all_blinks = 0
    all_no_blinks = 0
    shortest_blink = 60
    longest_blink = 0
    dropped_blinks = 0

    annotations_0 = []
    annotations_1 = []

    minall_r = 10
    maxall_r = 0

    minall_b = 10
    maxall_b = 0

    minall_g = 10
    maxall_g = 0

    minall_lids = 10
    maxall_lids = 0

    minall_pupil2corner = 20
    maxall_pupil2corner = 0

    minall_irisDiameter = 20
    maxall_irisDiameter = 0

    # video paths
    annds_paths = []
    for root,dirs, files in os.walk(dataset):
        for dir in files:
            name, ext = os.path.splitext(dir)
            if cfgs.dataset!='BlinkingValidationSetVideos':
                if ext in [".tag"]:
                    annds_paths.append(os.path.join(root,dir))
            elif ext in [".mp4"]:
                annds_paths.append(os.path.join(root,dir))

    vid_progress = tqdm.tqdm(annds_paths, total=len(annds_paths), desc="participants")
    for anns_path in vid_progress:
        video_name = os.path.dirname(anns_path)
        video_name = os.path.relpath(video_name, dataset)
        vid_progress.set_postfix(vid=video_name)

        # read annotations
        if cfgs.dataset=="BlinkingValidationSetVideos":
            blinking_anns = get_blinking_annotation(video_name)
        else:
            closeness_list, blinking_anns = read_annotations_tag(anns_path)
        # read data
        signals_path = os.path.join(dataset_root,"long_sequence", cfgs.dataset, video_name, "signals")
        std_path = os.path.join(signals_path, "stds.pkl")
        eyelids_path = os.path.join(signals_path, "eyelids_dists.pkl")

        iris_file_path = os.path.join(signals_path, "iris_diameter.pkl")
        pupil2corner_file_path = os.path.join(signals_path, "pupil2corner.pkl")

        faces_not_found_path = os.path.join(signals_path, "face_not_found.pkl")
        yaw_path = os.path.join(signals_path, "yaw_angles.pkl")
        pitch_path = os.path.join(signals_path, "pitch_angles.pkl")
        #
        with open(std_path, "rb") as f:
            std = pickle.load(f)['best']
        with open(eyelids_path, "rb") as f:
            eyelids_dist = pickle.load(f)['best']

        with open(iris_file_path, "rb") as f:
            iris_diameters = pickle.load(f)['best']
        with open(pupil2corner_file_path, "rb") as f:
            pupil2corners = pickle.load(f)['best']

        with open(faces_not_found_path, "rb") as f:
            faces_not_found = pickle.load(f)
        with open(yaw_path, "rb") as f:
            yaw_angles = pickle.load(f)
        with open(pitch_path, "rb") as f:
            pitch_angles = pickle.load(f)

        # the 4 required signals
        std_rgb = np.array(std)
        std_r =std_rgb[:,0].tolist()
        std_g =std_rgb[:,1].tolist()
        std_b =std_rgb[:,2].tolist()
        minall_r = min((minall_r,min(std_r)))
        maxall_r = max((maxall_r,max(std_r)))

        minall_b = min((minall_b,min(std_b)))
        maxall_b = max((maxall_b,max(std_b)))

        minall_g = min((minall_g,min(std_g)))
        maxall_g = max((maxall_g,max(std_g)))

        minall_lids = min((minall_lids,min(eyelids_dist)))
        maxall_lids = max((maxall_lids,max(eyelids_dist)))

        minall_irisDiameter = min((minall_irisDiameter,min(iris_diameters)))
        maxall_irisDiameter = max((maxall_irisDiameter,max(iris_diameters)))

        minall_pupil2corner = min((minall_pupil2corner,min(pupil2corners)))
        maxall_pupil2corner = max((maxall_pupil2corner,max(pupil2corners)))

        # convert to interval
        face_found_anns = get_intervals(faces_not_found, val=0)
        yaw_preds = get_intervals_between(yaw_angles, val=cfgs.yaw_range)
        pitch_preds = get_intervals_between(pitch_angles, val=cfgs.pitch_range)

        # annotations customization
        blinking_anns = blinking_anns.intersect(yaw_preds)
        blinking_anns = blinking_anns.intersect(pitch_preds)

        _noblink_range = [0, None]
        _once_legend = True

        for _blink in tqdm.tqdm(blinking_anns, total=len(blinking_anns), desc="blinks"):
            blink_length = _blink.stop - _blink.start+1

            # # # # # # # # # # # # # # # # # # # #
            # Generate True Negative -- no-blinks #
            # # # # # # # # # # # # # # # # # # # #
            if _blink.start - _noblink_range[0] > 30:
                _noblink_range[1] = _blink.start-1
                # generate valid intervals
                _from = _noblink_range[0]
                _to = _noblink_range[1]
                _interval = AnnotationOfIntervals(Unit.INDEX, [Interval(_from, _to)])
                if cfgs.face_found:
                    _valid_intervals = face_found_anns.intersect(_interval)
                else:
                    _valid_intervals = _interval
                
                _valid_intervals = _valid_intervals.intersect(yaw_preds)
                _valid_intervals = _valid_intervals.intersect(pitch_preds)
                
                # walk through valid intervals
                for _interval in _valid_intervals:
                    #
                    if _interval.length+1 < 30:
                        continue
                    #
                    _n_frames_skipped = 30 - cfgs.overlap
                    _max_examples = floor((_interval.length+1)/_n_frames_skipped)
                    _n_required_examples = int(_max_examples)
                    # check if #no-blink is required
                    if cfgs.no_blink is not None: 
                        _n_required_examples = cfgs.no_blink if cfgs.no_blink < _max_examples else _max_examples

                    random_start = _interval.start
                    _examples = 0

                    # generate no blinks in an interval
                    while _examples < _n_required_examples:

                        random_end = random_start+30
                        if random_end >= _blink.stop:
                            break

                        _interval_tmp = blinking_anns.intersect(AnnotationOfIntervals(Unit.INDEX, [Interval(random_start, random_end)]))
                        _partial_length = _interval_tmp.sumIntervalLengths

                        y_eyelids_noblink, _no_blink, _num_noblink = resample_noblink(y_in=eyelids_dist, start=random_start, stop=random_end-1, samples=30)
                        y_std_r_noblink, _, _num_noblink = resample_noblink(y_in=std_r, start=random_start, stop=random_end-1, samples=_num_noblink)
                        y_std_g_noblink, _, _num_noblink = resample_noblink(y_in=std_g, start=random_start, stop=random_end-1, samples=_num_noblink)
                        y_std_b_noblink, _, _num_noblink = resample_noblink(y_in=std_b, start=random_start, stop=random_end-1, samples=_num_noblink)
                        #
                        y_iris_diameters_noblink, _, _num_noblink = resample_noblink(y_in=iris_diameters, start=random_start, stop=random_end-1, samples=_num_noblink)
                        y_pupil2corners_noblink, _, _num_noblink = resample_noblink(y_in=pupil2corners, start=random_start, stop=random_end-1, samples=_num_noblink)
                        # new
                        y_yaws, _, _num_noblink = resample_noblink(y_in=yaw_angles, start=random_start, stop=random_end-1, samples=_num_noblink)
                        y_pitchs, _, _num_noblink = resample_noblink(y_in=pitch_angles, start=random_start, stop=random_end-1, samples=_num_noblink)

                        # save annotation
                        _ann = {
                            "pid": video_name,
                            "range": f"{random_start}-{random_end}",
                            "eyelids_dist": y_eyelids_noblink.tolist(), 
                            "iris_diameter": y_iris_diameters_noblink.tolist(),
                            "pupil2corner": y_pupil2corners_noblink.tolist(),
                            "std_r": y_std_r_noblink.tolist(), 
                            "std_g": y_std_g_noblink.tolist(), 
                            "std_b": y_std_b_noblink.tolist(),
                            "yaws": y_yaws.tolist(),
                            "pitchs": y_pitchs.tolist(),
                            "is_blink": 0,
                            "blink_length": int(_partial_length),
                            "start": 0,
                            "end": 0}

                        # plots
                        if cfgs.generate_plots:
                            fig, _ = plt.subplots(1, 1)
                            plt.title(f"{random_start}-{random_end}")
                            plt.plot(y_eyelids_noblink, "k", label='eyelids distance')
                            plt.plot(y_iris_diameters_noblink, "y", label='iris diameter')
                            # plt.plot(y_std_r_noblink, "r", label='std_r (Red)')
                            # plt.plot(y_std_g_noblink, "g", label='std_g (Green)')
                            # plt.plot(y_std_b_noblink, "b", label='std_b (Blue)')
                            plt.plot(_no_blink, label="No Blink")
                            plt.legend()
                            plt.tight_layout()
                            
                            img_output_path = os.path.join(output_folder, f"{video_name.replace('/','-')}_[{random_start}-{random_end}]_0.png")
                            plt.savefig(img_output_path, dpi=300, bbox_inches='tight')
                            plt.close(fig)

                        # increment noblinks
                        all_no_blinks+=1
                        # increment examples
                        _examples+=1
                        #
                        random_start += _n_frames_skipped
                        #
                        if int(_partial_length) > 0: continue
                        annotations_0.append(_ann)

            _noblink_range[0] = _blink.stop+1

            # # # # # # # # # # # # # # # # # # # # # #
            # Generate Blinks -- 6 augmented versions #
            # # # # # # # # # # # # # # # # # # # # # #
            if blink_length > 23:
                dropped_blinks+=1
                continue
            all_blinks += 1
            if blink_length > longest_blink:
                longest_blink = blink_length
            if blink_length < shortest_blink:
                shortest_blink = blink_length
            # # # # # # # #
            #  original   #
            # # # # # # # #
            try:
                y_eyelids, blink, _num = upsample_blink(y_in=eyelids_dist, start=_blink.start, stop=_blink.stop, samples=blink_length)
            except Exception as e:
                continue
            y_std_r, _, _num = upsample_blink(y_in=std_r, start=_blink.start, stop=_blink.stop, samples=blink_length)
            y_std_g, _, _num = upsample_blink(y_in=std_g, start=_blink.start, stop=_blink.stop, samples=blink_length)
            y_std_b, _, _num = upsample_blink(y_in=std_b, start=_blink.start, stop=_blink.stop, samples=blink_length)
            #
            y_iris_diameters_blink, _, _num = upsample_blink(y_in=iris_diameters, start=_blink.start, stop=_blink.stop, samples=blink_length)
            y_pupil2corners_blink, _, _num = upsample_blink(y_in=pupil2corners, start=_blink.start, stop=_blink.stop, samples=blink_length)
            # new
            y_yaws, _, _num = upsample_blink(y_in=yaw_angles, start=_blink.start, stop=_blink.stop, samples=blink_length)
            y_pitchs, _, _num = upsample_blink(y_in=pitch_angles, start=_blink.start, stop=_blink.stop, samples=blink_length)
            # blink interval
            extended_blink_interval = get_intervals(blink.tolist(), val=1)[0]
            # max_left_shift, max_right_shift = extended_blink_interval.start-2, 30-extended_blink_interval.stop-2
            half_width = (extended_blink_interval.length+1)/2
            max_left_shift, max_right_shift = 15-floor(half_width)-2, 15- ceil(half_width)-2

            # save annotation
            _ann = {
                "pid": video_name,
                "range": f"{_blink.start}-{_blink.stop}",
                "eyelids_dist": y_eyelids[15:-15].tolist(),
                "iris_diameter": y_iris_diameters_blink[15:-15].tolist(),
                "pupil2corner": y_pupil2corners_blink[15:-15].tolist(),
                "std_r": y_std_r[15:-15].tolist(), 
                "std_g": y_std_g[15:-15].tolist(), 
                "std_b": y_std_b[15:-15].tolist(),
                "yaws": y_yaws[15:-15].tolist(),
                "pitchs": y_pitchs[15:-15].tolist(),
                "is_blink": 1,
                "blink_length": int(blink_length),
                "start": int(extended_blink_interval.start-15),
                "end": int(extended_blink_interval.stop-15)}
            annotations_1.append(_ann)

            if not cfgs.eval:
                # # # # # #
                # Shifted #
                # # # # # #
                y_shifted_eyelids, _steps_original = shift(y_eyelids, [-max_left_shift, max_right_shift])
                #
                y_shifted_iris_diameter, _ = shift(y_iris_diameters_blink, _steps_original)
                y_shifted_pupil2corner, _ = shift(y_pupil2corners_blink, _steps_original)
                #
                y_shifted_std_r, _ = shift(y_std_r, _steps_original)
                y_shifted_std_g, _ = shift(y_std_g, _steps_original)
                y_shifted_std_b, _ = shift(y_std_b, _steps_original)
                blink_shifted, _ = shift(blink, _steps_original)
                extended_blink_interval = get_intervals(blink_shifted.tolist(), val=1)[0]
                # save annotation
                _ann = {
                    "pid": video_name,
                    "range": f"{_blink.start}-{_blink.stop}",
                    "eyelids_dist": y_shifted_eyelids[15:-15].tolist(),
                    #
                    "iris_diameter": y_shifted_iris_diameter[15:-15].tolist(),
                    "pupil2corner": y_shifted_pupil2corner[15:-15].tolist(),
                    #
                    "std_r": y_shifted_std_r[15:-15].tolist(),
                    "std_g": y_shifted_std_g[15:-15].tolist(), 
                    "std_b": y_shifted_std_b[15:-15].tolist(),
                    "yaws": y_yaws[15:-15].tolist(),
                    "pitchs": y_pitchs[15:-15].tolist(),
                    "is_blink": 1,
                    "blink_length": int(blink_length),
                    "start": int(extended_blink_interval.start-15),
                    "end": int(extended_blink_interval.stop-15)}
                annotations_1.append(_ann)
                # # # # # # #
                # Upsampled #
                # # # # # # #
                y_extended_eyelids, blink_extended, _num_extended = upsample_blink(y_in=eyelids_dist, start=_blink.start, stop=_blink.stop, samples=None)
                #
                y_extended_iris_diameters, _, _ = upsample_blink(y_in=iris_diameters, start=_blink.start, stop=_blink.stop, samples=_num_extended)
                y_extended_pupil2corners, _, _ = upsample_blink(y_in=pupil2corners, start=_blink.start, stop=_blink.stop, samples=_num_extended)
                #
                y_extended_std_r, _, _ = upsample_blink(y_in=std_r, start=_blink.start, stop=_blink.stop, samples=_num_extended)
                y_extended_std_g, _, _ = upsample_blink(y_in=std_g, start=_blink.start, stop=_blink.stop, samples=_num_extended)
                y_extended_std_b, _, _ = upsample_blink(y_in=std_b, start=_blink.start, stop=_blink.stop, samples=_num_extended)
                # blink interval
                extended_blink_interval = get_intervals(blink_extended.tolist(), val=1)[0]
                # max_left_shift, max_right_shift = extended_blink_interval.start-2, 30-extended_blink_interval.stop-2
                half_width = (extended_blink_interval.length+1)/2
                max_left_shift, max_right_shift = 15-floor(half_width)-2, 15- ceil(half_width)-2
                # save annotation
                _ann = {
                    "pid": video_name,
                    "range": f"{_blink.start}-{_blink.stop}",
                    "eyelids_dist": y_extended_eyelids[15:-15].tolist(),
                    #
                    "iris_diameter": y_extended_iris_diameters[15:-15].tolist(),
                    "pupil2corner": y_extended_pupil2corners[15:-15].tolist(),
                    #
                    "std_r": y_extended_std_r[15:-15].tolist(), 
                    "std_g": y_extended_std_g[15:-15].tolist(), 
                    "std_b": y_extended_std_b[15:-15].tolist(),
                    "yaws": y_yaws[15:-15].tolist(),
                    "pitchs": y_pitchs[15:-15].tolist(),
                    "is_blink": 1,
                    "blink_length": int(_num_extended),
                    "start": int(extended_blink_interval.start-15),
                    "end": int(extended_blink_interval.stop-15)}
                annotations_1.append(_ann)

                # # # # # # # # # # #
                # Upsampled Shifted #
                # # # # # # # # # # #
                y_extended_shifted_eyelids, _steps_extended = shift(y_extended_eyelids, [-max_left_shift, max_right_shift])
                #
                y_extended_shifted_iris_diameters, _ = shift(y_extended_iris_diameters, _steps_extended)
                y_extended_shifted_pupil2corners, _ = shift(y_extended_pupil2corners, _steps_extended)
                #
                y_extended_shifted_std_r, _ = shift(y_extended_std_r, _steps_extended)
                y_extended_shifted_std_g, _ = shift(y_extended_std_g, _steps_extended)
                y_extended_shifted_std_b, _ = shift(y_extended_std_b, _steps_extended)
                blink_extended_shifted, _ = shift(blink_extended, _steps_extended)
                # blink interval
                extended_blink_interval = get_intervals(blink_extended_shifted.tolist(), val=1)[0]
                # save annotation
                _ann = {
                    "pid": video_name,
                    "range": f"{_blink.start}-{_blink.stop}",
                    "eyelids_dist": y_extended_shifted_eyelids[15:-15].tolist(),
                    #
                    "iris_diameter": y_extended_shifted_iris_diameters[15:-15].tolist(),
                    "pupil2corner": y_extended_shifted_pupil2corners[15:-15].tolist(),
                    #
                    "std_r": y_extended_shifted_std_r[15:-15].tolist(), 
                    "std_g": y_extended_shifted_std_g[15:-15].tolist(), 
                    "std_b": y_extended_shifted_std_b[15:-15].tolist(),
                    "yaws": y_yaws[15:-15].tolist(),
                    "pitchs": y_pitchs[15:-15].tolist(),
                    "is_blink": 1,
                    "blink_length": int(_num_extended),
                    "start": int(extended_blink_interval.start-15),
                    "end": int(extended_blink_interval.stop-15)}
                annotations_1.append(_ann)

                # # # # # # # #
                # Downsampled #
                # # # # # # # #
                y_downsampled_eyelids, blink_downsampled, _num_shrinked = downsample_blink(y_in=eyelids_dist, start=_blink.start, stop=_blink.stop, samples=None)
                #
                y_downsampled_iris_diameters, _, _ = downsample_blink(y_in=iris_diameters, start=_blink.start, stop=_blink.stop, samples=_num_shrinked)
                y_downsampled_pupil2corners, _, _ = downsample_blink(y_in=pupil2corners, start=_blink.start, stop=_blink.stop, samples=_num_shrinked)
                #
                y_downsampled_std_r, _, _ = downsample_blink(y_in=std_r, start=_blink.start, stop=_blink.stop, samples=_num_shrinked)
                y_downsampled_std_g, _, _ = downsample_blink(y_in=std_g, start=_blink.start, stop=_blink.stop, samples=_num_shrinked)
                y_downsampled_std_b, _, _ = downsample_blink(y_in=std_b, start=_blink.start, stop=_blink.stop, samples=_num_shrinked)
                # blink interval
                downsampled_blink_interval = get_intervals(blink_downsampled.tolist(), val=1)[0]
                # max_left_shift, max_right_shift = downsampled_blink_interval.start-2, 30-downsampled_blink_interval.stop-2
                half_width = (downsampled_blink_interval.length+1)/2
                max_left_shift, max_right_shift = 15-floor(half_width)-2, 15- ceil(half_width)-2
                # save annotation
                _ann = {
                    "pid": video_name,
                    "range": f"{_blink.start}-{_blink.stop}",
                    "eyelids_dist": y_downsampled_eyelids[15:-15].tolist(),
                    #
                    "iris_diameter": y_downsampled_iris_diameters[15:-15].tolist(),
                    "pupil2corner": y_downsampled_pupil2corners[15:-15].tolist(),
                    #
                    "std_r": y_downsampled_std_r[15:-15].tolist(), 
                    "std_g": y_downsampled_std_g[15:-15].tolist(), 
                    "std_b": y_downsampled_std_b[15:-15].tolist(),
                    "yaws": y_yaws[15:-15].tolist(),
                    "pitchs": y_pitchs[15:-15].tolist(),
                    "is_blink": 1,
                    "blink_length": int(_num_shrinked),
                    "start": int(downsampled_blink_interval.start-15),
                    "end": int(downsampled_blink_interval.stop-15)}
                annotations_1.append(_ann)

                # # # # # # # # # # # #
                # Downsampled Shifted #
                # # # # # # # # # # # #
                y_downsampled_shifted_eyelids, _steps_downsampled = shift(y_downsampled_eyelids, [-max_left_shift, max_right_shift])
                #
                y_downsampled_shifted_iris_diameters, _ = shift(y_downsampled_iris_diameters, _steps_downsampled)
                y_downsampled_shifted_pupil2corners, _ = shift(y_downsampled_pupil2corners, _steps_downsampled)
                #
                y_downsampled_shifted_std_r, _ = shift(y_downsampled_std_r, _steps_downsampled)
                y_downsampled_shifted_std_g, _ = shift(y_downsampled_std_g, _steps_downsampled)
                y_downsampled_shifted_std_b, _ = shift(y_downsampled_std_b, _steps_downsampled)
                blink_downsampled_shifted, _ = shift(blink_downsampled, _steps_downsampled)

                downsampled_blink_interval = get_intervals(blink_downsampled_shifted.tolist(), val=1)[0]
                # save annotation
                _ann = {
                    "pid": video_name,
                    "range": f"{_blink.start}-{_blink.stop}",
                    "eyelids_dist": y_downsampled_shifted_eyelids[15:-15].tolist(),
                    #
                    "iris_diameter": y_downsampled_shifted_iris_diameters[15:-15].tolist(),
                    "pupil2corner": y_downsampled_shifted_pupil2corners[15:-15].tolist(),
                    #
                    "std_r": y_downsampled_shifted_std_r[15:-15].tolist(), 
                    "std_g": y_downsampled_shifted_std_g[15:-15].tolist(), 
                    "std_b": y_downsampled_shifted_std_b[15:-15].tolist(),
                    "yaws": y_yaws[15:-15].tolist(),
                    "pitchs": y_pitchs[15:-15].tolist(),
                    "is_blink": 1,
                    "blink_length": int(_num_shrinked),
                    "start": int(downsampled_blink_interval.start-15),
                    "end": int(downsampled_blink_interval.stop-15)}
                annotations_1.append(_ann)

            # plots
            if cfgs.generate_plots:
                
                # TODO:
                #   - fix plot size
                if not cfgs.eval:
                    # plotting
                    fig, _ = plt.subplots(2, 3)
                    # original
                    plt.subplot(2,3, 1)
                    plt.title(f"original len: {blink_length}")
                    plt.plot(y_eyelids[15:-15], "k", label='eyelids distance')
                    plt.plot(y_iris_diameters_blink[15:-15], "y", label='iris diameter')
                    plt.plot(y_eyelids[15:-15]/y_iris_diameters_blink[15:-15], "c", label='normalized')
                    # plt.plot(y_std_r[15:-15], "r", label='std_r (Red)')
                    # plt.plot(y_std_g[15:-15], "g", label='std_g (Green)')
                    # plt.plot(y_std_b[15:-15], "b", label='std_b (Blue)')
                    plt.plot(blink[15:-15], label="Blink")
                    # shifted
                    plt.subplot(2,3, 4)
                    plt.title(f"shifted by {_steps_original}")
                    plt.plot(y_shifted_eyelids[15:-15], "k")
                    plt.plot(y_shifted_iris_diameter[15:-15], "y", label='iris diameter')
                    plt.plot(y_shifted_eyelids[15:-15]/y_shifted_iris_diameter[15:-15], "c", label='normalized')
                    # plt.plot(y_shifted_std_r[15:-15], "r")
                    # plt.plot(y_shifted_std_g[15:-15], "g")
                    # plt.plot(y_shifted_std_b[15:-15], "b")
                    plt.plot(blink_shifted[15:-15])
                    # extended
                    plt.subplot(2,3, 2)
                    plt.title(f"extended {_num_extended}")
                    plt.plot(y_extended_eyelids[15:-15], "k")
                    plt.plot(y_extended_iris_diameters[15:-15], "y", label='iris diameter')
                    plt.plot(y_extended_eyelids[15:-15]/y_extended_iris_diameters[15:-15], "c", label='normalized')
                    # plt.plot(y_extended_std_r[15:-15], "r")
                    # plt.plot(y_extended_std_g[15:-15], "g")
                    # plt.plot(y_extended_std_b[15:-15], "b")
                    plt.plot(blink_extended[15:-15])
                    plt.subplot(2,3, 5)
                    #  extended shifted
                    plt.title(f"shifted by {_steps_extended}")
                    plt.plot(y_extended_shifted_eyelids[15:-15], "k")
                    plt.plot( y_extended_shifted_iris_diameters[15:-15], "y", label='iris diameter')
                    plt.plot(y_extended_shifted_eyelids[15:-15]/ y_extended_shifted_iris_diameters[15:-15], "c", label='normalized')
                    # plt.plot(y_extended_shifted_std_r[15:-15], "r")
                    # plt.plot(y_extended_shifted_std_g[15:-15], "g")
                    # plt.plot(y_extended_shifted_std_b[15:-15], "b")
                    plt.plot(blink_extended_shifted[15:-15])
                    # shrinked
                    plt.subplot(2,3, 3)
                    plt.title(f"shrunk {_num_shrinked}")
                    plt.plot(y_downsampled_eyelids[15:-15], "k")
                    plt.plot(y_downsampled_iris_diameters[15:-15], "y", label='iris diameter')
                    plt.plot(y_downsampled_eyelids[15:-15]/y_downsampled_iris_diameters[15:-15], "c", label='normalized')
                    # plt.plot(y_downsampled_std_r[15:-15], "r")
                    # plt.plot(y_downsampled_std_g[15:-15], "g")
                    # plt.plot(y_downsampled_std_b[15:-15], "b")
                    plt.plot(blink_downsampled[15:-15])
                    plt.subplot(2,3, 6)
                    #  extended shifted
                    plt.title(f"shifted by {_steps_downsampled}")
                    plt.plot(y_downsampled_shifted_eyelids[15:-15], "k")
                    plt.plot(y_downsampled_shifted_iris_diameters[15:-15], "y", label='iris diameter')
                    plt.plot(y_downsampled_shifted_eyelids[15:-15]/y_downsampled_shifted_iris_diameters[15:-15], "c", label='normalized')
                    # plt.plot(y_downsampled_shifted_std_r[15:-15], "r")
                    # plt.plot(y_downsampled_shifted_std_g[15:-15], "g")
                    # plt.plot(y_downsampled_shifted_std_b[15:-15], "b")
                    plt.plot(blink_downsampled_shifted[15:-15])
                
                else:
                    fig, _ = plt.subplots(1,1)
                    # original
                    plt.subplot(1,1, 1)
                    plt.title(f"original len: {blink_length}")
                    plt.plot(y_eyelids[15:-15], "k", label='eyelids distance')
                    plt.plot(y_iris_diameters_blink[15:-15], "y", label='iris diameter')
                    plt.plot(y_eyelids[15:-15]/y_iris_diameters_blink[15:-15], "c", label='normalized')
                    # plt.plot(y_std_r[15:-15], "r", label='std_r (Red)')
                    # plt.plot(y_std_g[15:-15], "g", label='std_g (Green)')
                    # plt.plot(y_std_b[15:-15], "b", label='std_b (Blue)')
                    plt.plot(blink[15:-15], label="Blink")

                if _once_legend:
                    fig.legend()
                    _once_legend = False

                plt.tight_layout()
                img_output_path = os.path.join(output_folder, f"{video_name.replace('/','-')}_[{_blink.start}-{_blink.stop}]_1.png")
                plt.savefig(img_output_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
    
    vid_progress.close()
    
    if cfgs.ratio > 0 and len(annotations_0) > len(annotations_1):
        ratio = min(int(len(annotations_1) * cfgs.ratio), len(annotations_0))
        print("1*ratio:", int(len(annotations_1) * cfgs.ratio))
        print("0:",len(annotations_0))
        print("1:",len(annotations_1))
        equal_annotation_0 = random.sample(annotations_0, ratio)
        equal_annotation_1 = annotations_1
    elif cfgs.equal:
        _max_num = min(len(annotations_0), len(annotations_1))
        equal_annotation_0 = random.sample(annotations_0, _max_num)
        equal_annotation_1 = random.sample(annotations_1, _max_num)
    elif cfgs.zero:
        equal_annotation_0 = []
        equal_annotation_1 = annotations_1
    else:
        equal_annotation_0 = annotations_0
        equal_annotation_1 = annotations_1

    annotations = equal_annotation_0 + equal_annotation_1

    if len(annotations) == 0: exit("No Blinks are found")

    # save annotations
    annotations_folder_path = os.path.join(annotations_folder, f"annotations-{cfgs.suffix}.json")
    with open(annotations_folder_path, "w") as f:
        json.dump(annotations, f)

    # save meta
    cfgs_dict = vars(cfgs)
    cfgs_dict['all_blinks'] = len(equal_annotation_1)
    cfgs_dict['all_no_blinks'] = len(equal_annotation_0)
    cfgs_dict['shortest_blink'] = int(shortest_blink)
    cfgs_dict['longest_blink'] = int(longest_blink)

    with open(meta_file, "w") as f:
        json.dump(cfgs_dict, f)

    # Satatistics
    print(f"All blinks: {cfgs_dict['all_blinks']}")
    print(f"ALL no blinks: {cfgs_dict['all_no_blinks']}")
    print(f"shortest blink: {shortest_blink}")
    print(f"longest blink: {longest_blink}")

    print(f"std_r: min {minall_r}, max {maxall_r}")
    print(f"std_g: min {minall_g}, max {maxall_g}")
    print(f"std_b: min {minall_b}, max {maxall_b}")
    print(f"eyelids: min {minall_lids}, max {maxall_lids}")
    print(f"iris diameter: min {minall_irisDiameter}, max {maxall_irisDiameter}")
    print(f"pupil2corner: min {minall_pupil2corner}, max {maxall_pupil2corner}")


if __name__=="__main__":
    cfgs = parse()
    main(cfgs)
