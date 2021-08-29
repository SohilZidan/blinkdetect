#!/usr/bin/env python3
# coding: utf-8

from collections import namedtuple
import os
import argparse
import numpy as np
from numpy.core.defchararray import center
from numpy.lib.twodim_base import eye
import pandas as pd
import  cv2
import tqdm

import matplotlib.pyplot as plt


dataset_root = os.path.join(os.path.dirname(__file__), "..", "dataset")
cache_root = os.path.join(os.path.dirname(__file__),"..","cache")


def generate_mean_std_plot(img_path, stds, means, eyelids_dists, yaws, faces_not_found, bboxes, x_range, frame_num, maxes, mins, out_folder_path, show_face: bool=False):
    # read image
    img = cv2.imread(img_path)
    try:
    # if img.size == 0:
    #     print(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print("-------------------")
        print(img_path, type(img))
        print("----------------------")
        raise(e)
    return
    # face detection
    # if show_face:
    # bboxes: left, top, right, bottom
    # faces = RetinaFace.extract_faces(img_path = img)#, align = True)
    face = img[bboxes[1]:bboxes[3], bboxes[0]: bboxes[2]]#img np.full_like(a=img, fill_value=0)
    # print(f"{len(faces)} faces detected")
    if face.size == 0:
        face = np.full_like(a=img, fill_value=0)
    # else: print(f"No face is detected for {img_path}")

    # 


    fig = plt.figure()

    
    
    
    
    

    
    # plt.subplot(5, 1, 1)
    # plt.imshow(img)
    # plt.axis('image')

    # fig, axes = plt.subplots(nrows=5,ncols=1, sharex=True)

    # plt.subplot(5, 1, 1, sharex=True)
    ax1 = fig.add_subplot(5, 1, 1)
    plt.imshow(face)
    plt.axis('image')
    # ax1.set_xticklabels([], [])
    # ax1.set_yticklabels([], [])
    plt.axis('off')
    

    # plt.subplot(5, 1, 2, sharex=True)
    # plt.title("mean")
    ax2 = fig.add_subplot(5, 1, 2)
    plt.ylim(mins[0],maxes[0])
    # axes[1].ylim(mins[0], maxes[0])
    
    plt.bar(x_range, np.array(faces_not_found) * maxes[0], width=1, align='center', visible=False)
    plt.plot(x_range, means)#[:int(len(zi)/2)])
    plt.plot(x_range, means,'.')#[:int(len(zi)/2)])
    plt.axvline(x=frame_num, color='red', linestyle='--')
    # axes[1].ylabel('mean')
    plt.ylabel('mean')
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    # ax2.set_xticklabels([])

    # axes[2].subplot(5, 1, 3, sharex=True)
    # plt.title("standard deviation")
    # axes[2].ylim(mins[1], maxes[1])
    ax3 = fig.add_subplot(5, 1, 3, sharex = ax2)
    plt.ylim(mins[1],maxes[1])
    plt.bar(x_range, np.array(faces_not_found) * maxes[1], width=1, align='center')
    plt.plot(x_range, stds)
    plt.plot(x_range, stds, '.g')
    plt.axvline(x=frame_num, color='red', linestyle='--')
    # axes[2].ylabel('std')
    plt.ylabel('std')
    # ax3.set_xticklabels([])
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off

    # plt.subplot(5, 1, 4, sharex=True)
    # plt.title("upper-lower eyelids hausdorff distance")
    ax4 = fig.add_subplot(5, 1, 4, sharex = ax3)
    plt.bar(x_range, np.array(faces_not_found) * maxes[2], width=1, align='center')
    # axes[3].ylim(mins[2], maxes[2])
    plt.ylim(mins[2],maxes[2])
    plt.plot(x_range, eyelids_dists)
    plt.plot(x_range, eyelids_dists, '.y')
    plt.axvline(x=frame_num, color='red', linestyle='--')
    # axes[3].ylabel('eyelids dist')
    plt.ylabel('eyelids\ndist')
    # ax4.set_xticklabels([])
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    # axes[3].xlabel('frame')
    

    # plt.subplot(5, 1, 5, sharex=True)
    # plt.title("upper-lower eyelids hausdorff distance")
    ax5 = fig.add_subplot(5, 1, 5, sharex = ax4)
    plt.bar(x_range, np.array(faces_not_found) * maxes[3], width=1, align='center')
    # axes[4].ylim(mins[3], maxes[3])
    plt.ylim(mins[3],maxes[3])
    plt.plot(x_range, yaws)
    plt.plot(x_range, yaws, '.b')
    plt.axvline(x=frame_num, color='red', linestyle='--')
    # axes[4].ylabel('yaw angle')
    plt.ylabel('yaw\nangle')
    plt.xlabel('frame')
    

    # fig, axes = plt.subplots(nrows=3,ncols=1)
    # # img plot
    # axes[0].imshow(img)
    # axes[0].axis('image')

    # # mean plot
    # axes[1].set_title("mean change")
    # axes[1].plot(x_range, means)
    # axes[1].axvline(x=frame_num, color='red', linestyle='--')
    # # std plot
    # axes[2].set_title("standard deviation change")
    # axes[2].plot(x_range, stds)
    # axes[2].axvline(x=frame_num, color='red', linestyle='--')

    fig.tight_layout()
    
    # out_path = os.path.join(out_folder_path, f"{frame_num}.jpg")
    # out_path = os.path.join(out_folder_path, f"{frame_num}_{'face' if show_face else 'full'}_timeseries.jpg")#,f"{_ext}")
    out_path = os.path.join(out_folder_path, f"{frame_num}.jpg")
    # 
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    # exit()

def get_N_frames_centered(center_frame: int, frames: set, frames_num: int):
    """generates the longest consecutive series of frames around the `center_frame` in the `frames` list

    Args:
        center_frame (int): the center frame
        frames (set): frames available
        frames_num (int): number of frames before and after the center frame

    Returns:
        list: [description]
    """
    
    start_frame = 0 if frames_num > center_frame else center_frame-frames_num
    # last_start = start
    end_frame = center_frame+frames_num

    for i in range(start_frame,center_frame+1):
        if str(i) not in frames:
            start_frame=i+1

    for i in range(center_frame,center_frame+frames_num+1):
        if str(i) not in frames:
            end_frame=i-1
            break
    
    return [str(i) for i in range(start_frame, end_frame+1)]

def check_frames_range(start: int, end: int, frames: set) -> list:
    for i in range(start,end+1):
        if f"{i:06}" not in frames:
            return False

    return True

def get_suggested_ranges(frames: list):
    _ranges = []
    from itertools import groupby
    from operator import itemgetter
    for k, g in groupby(enumerate(frames), lambda ix : int(ix[0]) - int(ix[1])):
        _range = list(map(itemgetter(1), g))
        _ranges.append([_range[0], _range[-1]])
    
    return _ranges


def parser():
    _parser = argparse.ArgumentParser()
    _parser.add_argument('-pid', '--participant_id', required=True)
    _parser.add_argument('-rng', '--range', type=int, default=30)
    # _parser.add_argument('--show_face', action='store_true', help='if true it does not generate a new plot')
    # _parser.add_argument('--resume', action='store_true', help='if true it does not generate a new plot')
    _parser.add_argument('--generate_plots', action='store_true', help='')
    
    _parser.add_argument('--select_eye', type=str, choices=['right', 'left', 'best'], default='best')

    return _parser.parse_args()




def find_max_list(ls: list):
    new_ls = [[],[],[]]
    try:
        for _item in ls:
            if _item is None: continue
            for _dim_idx, _dim_val in enumerate(_item):
                if _dim_val is None: continue
                new_ls[_dim_idx].append(_dim_val)
        return [max(new_ls[0]), max(new_ls[0]), max(new_ls[0])]
    except Exception as e:
        print(_item, type(_item))
        raise(e)

def find_min_list(ls: list):
    new_ls = [[],[],[]]

    for _item in ls:
        if _item is None: continue
        for _dim_idx, _dim_val in enumerate(_item):
            if _dim_val is None: continue
            new_ls[_dim_idx].append(_dim_val)
    return [min(new_ls[0]), min(new_ls[0]), min(new_ls[0])]

def find_min(ls: list):
    if len(ls) > 0 and type(ls[0]) is list:
        return find_min_list(ls)
    new_ls = []
    for _item in ls:
        if _item is None: continue
        new_ls.append(_item)
    return min(new_ls)

def find_max(ls: list):
    try:
        new_ls = []
        if len(ls) > 0 and type(ls[0]) is list:
            return find_max_list(ls)
        for _item in ls:
            if _item is None: continue
            new_ls.append(_item)
        return max(new_ls)
    except Exception as e:
        print(ls)
        print(ls[0])
        print(type(ls), len(ls))
        print(ls[0],type(ls[0]))
        raise(e)

def fill_missing_frames(input_df: pd.DataFrame):
        data_df = input_df.copy()
        ordered_frames = data_df.index

        # fille the missing frames between min and max 
        _first_frame = int(min(ordered_frames))
        _last_frame = int(max(ordered_frames))
        for i in range(_first_frame, _last_frame+1):
            _current_frame = f'{i:06}'
            if _current_frame in ordered_frames: continue

            new_row = [
                participant_id, # 1
                'face_1', # 1
                f'{dataset_root}/BlinkingValidationSetVideos/{participant_id}/frames/{_current_frame}', # 1
                1,0, # 2
                0,0,0,0, # 4
                np.nan,np.nan, # 2
                np.nan,np.nan, # 2
                np.nan,np.nan, # 2
                np.nan,np.nan,np.nan, # 3
                np.nan,np.nan, # 2
                [np.nan,np.nan,np.nan],[np.nan,np.nan,np.nan], # 2
                [np.nan,np.nan,np.nan],[np.nan,np.nan,np.nan], # 2
                [0, 0, 0, 0, 0, 0], # 1
                [0, 0, 0, 0, 0, 0], # 1
            ]

            # add the missing row
            data_df.loc[_current_frame] = new_row
        
        return data_df.sort_index()

        # new_row = {
        #     'participant_id': participant_id,
        #     'face_id': 'face_1',
        #     'img_path': f'{dataset_root}/BlinkingValidationSetVideos/{participant_id}/frames/{_current_frame}',
        #     'faces_not_found': 1,
        #     'faces_number': 0,
        #     'left': None,
        #     'top': None,
        #     'right': None,
        #     'bottom': None,
        #     'left_eye_x': None,
        #     'left_eye_y': None,
        #     'right_eye_x': None,
        #     'right_eye_y': None,
        #     'nose_x': None,
        #     'nose_y': None,
        #     'yaw': None,
        #     'pitch': None,
        #     'roll': None,
        #     'eyelids_dist_right': None,
        #     'eyelids_dist_left': None,
        #     'mean_color_right': None,
        #     'mean_color_left': None,
        #     'std_left': None,
        #     'std_right': None,
        #     'line_points_left': [0, 0, 0, 0, 0, 0]
        # }


if __name__=="__main__":
    args = parser()
    participant_id = args.participant_id
    # start, end = args.range
    # resume = args.resume
    # show_face = args.show_face
    generate_plots = args.generate_plots
    selected_eye = args.select_eye
    _rng = args.range

    # input
    input_file_path_hdf5 = os.path.join(dataset_root,"eye_landmarks", f"{participant_id}","eyeinfo.hdf5")
    assert os.path.exists(input_file_path_hdf5), f"{input_file_path_hdf5} not existed"
    
    # output
    output_path = os.path.join(dataset_root,"tracked_faces", f"{participant_id}", "timeseries_plots")
    output_results_path = os.path.join(dataset_root,"tracked_faces", f"{participant_id}", "signals")

    os.makedirs(output_path, exist_ok=True)
    os.makedirs(output_results_path, exist_ok=True)
    # results files paths
    means_file_path = os.path.join(output_results_path, "means.pkl")
    
    stds_file_path = os.path.join(output_results_path, "stds.pkl")
    
    eyelids_file_path = os.path.join(output_results_path, "eyelids_dists.pkl")
    
    yaws_file_path = os.path.join(output_results_path, "yaw_angles.pkl")
    
    pitch_file_path = os.path.join(output_results_path, "pitch_angles.pkl")
    
    faces_not_found_file_path = os.path.join(output_results_path, "face_not_found.pkl")
    

    
    with pd.HDFStore(input_file_path_hdf5) as store:
        data_df = store['eyes_info_dataset_01']
        metadata = store.get_storer('eyes_info_dataset_01').attrs.metadata

    # reindexing
    data_df = data_df.reset_index(level=['participant_id', 'face_id'])
    
    

    # fill missing frames
    _frames = fill_missing_frames(data_df)

    # type casting
    data_types_dict = {
            'participant_id': str,
            'face_id': str,
            'img_path': str,
            'faces_not_found': int,
            'faces_number': int,
            'left': int,
            'top': int,
            'right': int,
            'bottom': int,
            'left_eye_x': np.float64,
            'left_eye_y': np.float64,
            'right_eye_x': np.float64,
            'right_eye_y': np.float64,
            'nose_x': np.float64,
            'nose_y': np.float64,
            'yaw': np.float64,
            'pitch': np.float64,
            'roll': np.float64,
            'eyelids_dist_right': np.float64,
            'eyelids_dist_left': np.float64,
            'mean_color_right': object,
            'mean_color_left': object,
            'std_left': object,
            'std_right': object,
            # 'line_points_left': object
        }
    _frames = _frames.astype(data_types_dict)

    # # check frames range --> must be continuous
    # if not check_frames_range(start, end, ordered_frames):
    #     _ranges = get_suggested_ranges(ordered_frames)
    #     exit(f"range entered is not available, suggested ranges:\n{_ranges}")

    ordered_frames = sorted(_frames.index)
    

    max_mean_left = find_max(_frames['mean_color_left'].tolist())
    max_mean_right = find_max(_frames['mean_color_right'].tolist())
    max_std_left = find_max(_frames['std_left'].tolist())
    max_std_right = find_max(_frames['std_right'].tolist())
    max_dist_right = find_max(_frames['eyelids_dist_right'].tolist())
    max_dist_left = find_max(_frames['eyelids_dist_left'].tolist())
    max_yaw = find_max(_frames['yaw'].tolist())
    min_yaw = find_min(_frames['yaw'].tolist())
    max_pitch = find_max(_frames['pitch'].tolist())
    min_pitch = find_min(_frames['pitch'].tolist())
    
    # if yaw > 0 right looking :take right eye
    # left eye means the eye closer to the left of the screen 
    # else left-looking: take left eye

    # Find non values
    # TODO: handle the face not found
    # non_idxes = []
    # for _idx, _item in enumerate(_frames['mean_color'].tolist()):
    #     if _item is None: 
    #         non_idxes.append(_idx)


    # eye_options = ["right"]
    _means = {"right": [], "left": [], "best": []}
    _stds = {"right": [], "left": [], "best": []}
    _eyelids_dists = {"right": [], "left": [], "best": []}

    _paths = []
    _yaws = []
    _pitchs = []
    _bboxes = []
    _faces_not_found = []

    # print(_frames.index)
    # print(_frames.columns)
    for idx, _frame in enumerate(ordered_frames):
        _paths.append(_frames.loc[_frame]["img_path"])
        
        _bboxes.append(list(_frames.loc[_frame][['left', 'top', 'right', 'bottom']].astype(np.int32)))
        _faces_not_found.append(_frames.loc[_frame]["faces_not_found"])

        _yaws.append(0 if _frames.loc[_frame]["faces_not_found"]==1 else _frames.loc[_frame]["yaw"])
        _pitchs.append(0 if _frames.loc[_frame]["faces_not_found"]==1 else _frames.loc[_frame]["pitch"])

        _means["right"].append([0, 0,0] if _frames.loc[_frame]["faces_not_found"]==1 else _frames.loc[_frame]["mean_color_right"])
        _means["left"].append([0, 0,0] if _frames.loc[_frame]["faces_not_found"]==1 else _frames.loc[_frame]["mean_color_left"])
        
        _stds["right"].append([0, 0,0] if _frames.loc[_frame]["faces_not_found"]==1 else _frames.loc[_frame]["std_right"])
        _stds["left"].append([0, 0,0] if _frames.loc[_frame]["faces_not_found"]==1 else _frames.loc[_frame]["std_left"])

        _eyelids_dists["right"].append(0 if _frames.loc[_frame]["faces_not_found"]==1 else _frames.loc[_frame]["eyelids_dist_right"])
        _eyelids_dists["left"].append(0 if _frames.loc[_frame]["faces_not_found"]==1 else _frames.loc[_frame]["eyelids_dist_left"])

        if _frames.loc[_frame]["faces_not_found"]==1:
            _means["best"].append([0,0,0])
            _stds["best"].append([0,0,0])
            _eyelids_dists["best"].append(0)
        elif _frames.loc[_frame]['yaw'] > 0:
            _means["best"].append(_frames.loc[_frame]["mean_color_right"])
            _stds["best"].append(_frames.loc[_frame]["std_right"])
            _eyelids_dists["best"].append(_frames.loc[_frame]["eyelids_dist_right"])
        else:
            _means["best"].append(_frames.loc[_frame]["mean_color_left"])
            _stds["best"].append(_frames.loc[_frame]["std_left"])
            _eyelids_dists["best"].append(_frames.loc[_frame]["eyelids_dist_left"])

        
    # save signals
    import pickle
    with open(means_file_path, "wb" ) as pkl_file:
        pickle.dump(_means, pkl_file)
    with open(stds_file_path, "wb" ) as pkl_file:
        pickle.dump(_stds, pkl_file)
    with open(eyelids_file_path, "wb" ) as pkl_file:
        pickle.dump(_eyelids_dists, pkl_file)
    with open(yaws_file_path, "wb" ) as pkl_file:
        pickle.dump(_yaws, pkl_file)
    with open(pitch_file_path, "wb" ) as pkl_file:
        pickle.dump(_pitchs, pkl_file)
    with open(faces_not_found_file_path, "wb" ) as pkl_file:
        pickle.dump(_faces_not_found, pkl_file)

    print("signals are saved")


    # generate plots
    max_mean = find_max(_means[selected_eye])
    max_std = find_max(_stds[selected_eye])
    max_dist = find_max(_eyelids_dists[selected_eye])

    
    if generate_plots:
        for idx,_frame in tqdm.tqdm(enumerate(ordered_frames), total=len(ordered_frames)):
            _start = int(idx/_rng) * _rng
            _end = len(ordered_frames) if _start+_rng+1 > len(ordered_frames) else _start+_rng
            
            generate_mean_std_plot(
                img_path=_paths[idx], 
                stds=_stds[selected_eye][_start:_end], 
                means=_means[selected_eye][_start:_end], 
                eyelids_dists=_eyelids_dists[selected_eye][_start:_end],
                yaws=_yaws[_start:_end],
                faces_not_found=_faces_not_found[_start:_end],
                bboxes=_bboxes[idx],
                x_range=range(int(ordered_frames[_start]), int(ordered_frames[_end-1])+1), 
                frame_num=int(_frame),
                maxes = [max_mean, max_std, max_dist, max_yaw],
                mins = [0, 0, 0, min_yaw],
                out_folder_path=output_path,)
    else:
        print("No plots are generated, your work is done")

