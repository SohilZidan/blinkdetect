#!/usr/bin/env python3

import os
import argparse
import pickle
import tqdm
import numpy as np
import pandas as pd
import cv2

import matplotlib.pyplot as plt

dataset_root = os.path.join(os.path.dirname(__file__), "..", "dataset")


def generate_mean_std_plot(
        img_path,
        stds, means,
        eyelids_dists, iris_diameters,
        faces_not_found, bboxes,
        x_range, frame_num,
        maxes, mins,
        out_folder_path, show_face: bool = False):
    # read image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # face detection
    face = img[bboxes[1]:bboxes[3], bboxes[0]: bboxes[2]]

    if face.size == 0:
        face = np.full_like(a=img, fill_value=0)

    fig = plt.figure()

    # FACE
    ax1 = fig.add_subplot(2, 1, 1)
    plt.imshow(face)
    plt.axis('image')
    plt.axis('off')

    # MEAN
    # ax2 = fig.add_subplot(5, 1, 2)
    # plt.ylim(mins[0], max(maxes[0]))
    # plt.bar(x_range, np.array(faces_not_found) * max(maxes[0]), width=1, align='center', visible=False)
    # plt.plot(x_range, np.mean(means, axis=1))
    # plt.plot(x_range, np.mean(means, axis=1), '.')
    # plt.axvline(x=frame_num, color='red', linestyle='--')
    # plt.ylabel('mean')
    # plt.tick_params(
    #     axis='x',          # changes apply to the x-axis
    #     which='both',      # both major and minor ticks are affected
    #     bottom=False,      # ticks along the bottom edge are off
    #     top=False,         # ticks along the top edge are off
    #     labelbottom=False) # labels along the bottom edge are off

    # STD
    # ax3 = fig.add_subplot(5, 1, 3, sharex = ax2)
    # plt.ylim(mins[1], max(maxes[1]))
    # plt.bar(x_range, np.array(faces_not_found) * max(maxes[1]), width=1, align='center')
    # plt.plot(x_range, np.mean(stds, axis=1))
    # plt.plot(x_range, np.mean(stds, axis=1), '.g')
    # plt.axvline(x=frame_num, color='red', linestyle='--')
    # plt.ylabel('std')
    # plt.tick_params(
    #     axis='x',          # changes apply to the x-axis
    #     which='both',      # both major and minor ticks are affected
    #     bottom=False,      # ticks along the bottom edge are off
    #     top=False,         # ticks along the top edge are off
    #     labelbottom=False) # labels along the bottom edge are off

    # EYELDIDS DIST
    ax4 = fig.add_subplot(2, 1, 2)
    minall = min(mins[2], mins[3])
    maxall = max(maxes[2], maxes[3])
    plt.bar(x_range, np.array(faces_not_found)
            * maxes[2], width=1, align='center')
    plt.ylim(minall, maxall)
    plt.plot(x_range, eyelids_dists, 'y', label="eyelids distance")
    plt.plot(x_range, eyelids_dists, '.y')
    plt.axvline(x=frame_num, color='red', linestyle='--')
    plt.ylabel('pixel')
    plt.xlabel('frame')
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off

    # IRIS
    # plt.bar(x_range, np.array(faces_not_found) * maxes[3], width=1, align='center')
    # plt.ylim(minall, maxall)
    plt.plot(x_range, iris_diameters, 'b', label="iris diameter")
    plt.plot(x_range, iris_diameters, '.b')
    plt.grid()
    # plt.axvline(x=frame_num, color='red', linestyle='--')
    # plt.ylabel('iris diameter')
    # plt.xlabel('frame')
    plt.legend()

    fig.tight_layout()

    out_path = os.path.join(out_folder_path, f"{frame_num}.jpg")
    #
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def parser():
    _parser = argparse.ArgumentParser()
    _parser.add_argument(
        '--dataset', default="BlinkingValidationSetVideos",
        choices=["BlinkingValidationSetVideos", "eyeblink8", "talkingFace", "zju", "RN"])
    _parser.add_argument('-rng', '--range', type=int, default=30)
    _parser.add_argument('--generate_plots', action='store_true', help='')
    _parser.add_argument('--select_eye', type=str,
                         choices=['right', 'left', 'best'], default='best')
    return _parser.parse_args()


def find_max_list(ls: list):
    new_ls = [[], [], []]
    try:
        for _item in ls:
            if _item is None:
                continue
            for _dim_idx, _dim_val in enumerate(_item):
                if _dim_val is None:
                    continue
                new_ls[_dim_idx].append(_dim_val)
        return [max(new_ls[0]), max(new_ls[0]), max(new_ls[0])]
    except Exception as e:
        print(_item, type(_item))
        raise(e)


def find_min_list(ls: list):
    new_ls = [[], [], []]

    for _item in ls:
        if _item is None:
            continue
        for _dim_idx, _dim_val in enumerate(_item):
            if _dim_val is None:
                continue
            new_ls[_dim_idx].append(_dim_val)
    return [min(new_ls[0]), min(new_ls[0]), min(new_ls[0])]


def find_min(ls: list):
    if len(ls) > 0 and type(ls[0]) is list:
        return find_min_list(ls)
    new_ls = []
    for _item in ls:
        if _item is None:
            continue
        new_ls.append(_item)
    return min(new_ls)


def find_max(ls: list):
    try:
        new_ls = []
        if len(ls) > 0 and type(ls[0]) is list:
            return find_max_list(ls)
        for _item in ls:
            if _item is None:
                continue
            new_ls.append(_item)
        return max(new_ls)
    except Exception as e:
        print(ls)
        print(ls[0])
        print(type(ls), len(ls))
        print(ls[0], type(ls[0]))
        raise(e)


def fill_missing_frames(input_df: pd.DataFrame, participant_id, dataset):
    data_df = input_df.copy()
    ordered_frames = data_df.index

    # fille the missing frames between min and max
    _first_frame = int(min(ordered_frames))
    _last_frame = int(max(ordered_frames))
    for i in range(_first_frame, _last_frame+1):
        _current_frame = f'{i:06}'
        if _current_frame in ordered_frames:
            continue

        new_row = [
            participant_id,  # 1
            'face_1',  # 1
            f'{dataset}/{participant_id}/frames/{_current_frame}',  # 1
            1, 0,  # 2
            0, 0, 0, 0,  # 4
            np.nan, np.nan,  # 2
            np.nan, np.nan,  # 2
            np.nan, np.nan,  # 2
            np.nan, np.nan, np.nan,  # 3
            np.nan, np.nan,  # 2
            np.nan, np.nan,  # 2
            np.nan, np.nan,  # 2
            [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan],  # 2
            [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan],  # 2
            [0, 0, 0, 0, 0, 0],  # 1
            [0, 0, 0, 0, 0, 0],  # 1
        ]

        # add the missing row
        data_df.loc[_current_frame] = new_row

    return data_df.sort_index()


if __name__ == "__main__":
    args = parser()
    generate_plots = args.generate_plots
    selected_eye = args.select_eye
    _rng = args.range
    dataset = os.path.join(dataset_root, args.dataset)

    #
    # video paths
    videos_paths = []
    for root, dirs, files in os.walk(dataset):
        for dir in files:
            name, ext = os.path.splitext(dir)
            if ext in [".avi", ".mov", ".wmv", ".mp4"]:
                videos_paths.append(os.path.join(root, dir))

    for video_path in tqdm.tqdm(videos_paths, total=len(videos_paths), desc="video"):
        video_name = os.path.dirname(video_path)
        video_name = os.path.relpath(video_name, dataset)
        # input
        input_file_path_hdf5 = os.path.normpath(os.path.join(
            dataset_root, "eye_landmarks", args.dataset, video_name, "eyeinfo.hdf5"))
        if not os.path.exists(input_file_path_hdf5):
            print(f"{input_file_path_hdf5} not existed")
            continue

        # output
        output_path = os.path.normpath(os.path.join(
            dataset_root, "long_sequence", args.dataset, video_name, "timeseries_plots"))
        output_results_path = os.path.normpath(os.path.join(
            dataset_root, "long_sequence", args.dataset, video_name, "signals"))
        # results files paths
        means_file_path = os.path.join(output_results_path, "means.pkl")
        stds_file_path = os.path.join(output_results_path, "stds.pkl")
        eyelids_file_path = os.path.join(
            output_results_path, "eyelids_dists.pkl")

        iris_file_path = os.path.join(output_results_path, "iris_diameter.pkl")
        pupil2corner_file_path = os.path.join(
            output_results_path, "pupil2corner.pkl")

        yaws_file_path = os.path.join(output_results_path, "yaw_angles.pkl")
        pitch_file_path = os.path.join(output_results_path, "pitch_angles.pkl")
        faces_not_found_file_path = os.path.join(
            output_results_path, "face_not_found.pkl")

        os.makedirs(output_path, exist_ok=True)
        os.makedirs(output_results_path, exist_ok=True)

        with pd.HDFStore(input_file_path_hdf5) as store:
            data_df = store['eyes_info_dataset_01']
            # metadata = store.get_storer('eyes_info_dataset_01').attrs.metadata

        # reindexing
        data_df = data_df.reset_index(level=['participant_id', 'face_id'])

        # fill missing frames
        _frames = fill_missing_frames(data_df, video_name, dataset)

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

            "right_eyelids_dist": np.float64,
            "left_eyelids_dist": np.float64,

            "right_diameter": np.float64,
            "left_diameter": np.float64,

            "right_pupil2corner": np.float64,
            "left_pupil2corner": np.float64,

            'right_mean_color': object,
            'left_mean_color': object,

            'right_std_color': object,
            'left_std_color': object,
            # 'line_points_left': object
        }

        _frames = _frames.astype(data_types_dict)

        ordered_frames = sorted(_frames.index)

        # max and min value for each feature
        max_mean_right = find_max(_frames['right_mean_color'].tolist())
        max_mean_left = find_max(_frames['left_mean_color'].tolist())

        max_std_right = find_max(_frames['right_std_color'].tolist())
        max_std_left = find_max(_frames['left_std_color'].tolist())

        max_dist_right = find_max(_frames['right_eyelids_dist'].tolist())
        max_dist_left = find_max(_frames['left_eyelids_dist'].tolist())

        max_pupil2corner_right = find_max(
            _frames['right_pupil2corner'].tolist())
        max_pupil2corner_left = find_max(_frames['left_pupil2corner'].tolist())

        max_iris_right = find_max(_frames['right_diameter'].tolist())
        max_iris_left = find_max(_frames['left_diameter'].tolist())

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

        _means = {"right": [], "left": [], "best": []}
        _stds = {"right": [], "left": [], "best": []}
        _eyelids_dists = {"right": [], "left": [], "best": []}
        _diameters = {"right": [], "left": [], "best": []}
        _pupil2corners = {"right": [], "left": [], "best": []}

        _paths = []
        _yaws = []
        _pitchs = []
        _bboxes = []
        _faces_not_found = []

        for idx, _frame in tqdm.tqdm(enumerate(ordered_frames), total=len(ordered_frames), desc="frame"):
            _paths.append(_frames.loc[_frame]["img_path"])

            _bboxes.append(
                list(_frames.loc[_frame][['left', 'top', 'right', 'bottom']].astype(np.int32)))
            _faces_not_found.append(_frames.loc[_frame]["faces_not_found"])

            _yaws.append(
                0 if _frames.loc[_frame]["faces_not_found"] == 1 else _frames.loc[_frame]["yaw"])
            _pitchs.append(
                0 if _frames.loc[_frame]["faces_not_found"] == 1 else _frames.loc[_frame]["pitch"])

            _means["right"].append([0, 0, 0] if _frames.loc[_frame]["faces_not_found"]
                                   == 1 else _frames.loc[_frame]["right_mean_color"])
            _means["left"].append([0, 0, 0] if _frames.loc[_frame]["faces_not_found"]
                                  == 1 else _frames.loc[_frame]["left_mean_color"])

            _stds["right"].append([0, 0, 0] if _frames.loc[_frame]["faces_not_found"]
                                  == 1 else _frames.loc[_frame]["right_std_color"])
            _stds["left"].append([0, 0, 0] if _frames.loc[_frame]["faces_not_found"]
                                 == 1 else _frames.loc[_frame]["left_std_color"])

            _eyelids_dists["right"].append(
                0 if _frames.loc[_frame]["faces_not_found"] == 1 else _frames.loc[_frame]["right_eyelids_dist"])
            _eyelids_dists["left"].append(
                0 if _frames.loc[_frame]["faces_not_found"] == 1 else _frames.loc[_frame]["left_eyelids_dist"])

            _diameters["right"].append(
                0 if _frames.loc[_frame]["faces_not_found"] == 1 else _frames.loc[_frame]["right_diameter"])
            _diameters["left"].append(
                0 if _frames.loc[_frame]["faces_not_found"] == 1 else _frames.loc[_frame]["left_diameter"])

            _pupil2corners["right"].append(
                0 if _frames.loc[_frame]["faces_not_found"] == 1 else _frames.loc[_frame]["right_pupil2corner"])
            _pupil2corners["left"].append(
                0 if _frames.loc[_frame]["faces_not_found"] == 1 else _frames.loc[_frame]["left_pupil2corner"])

            if _frames.loc[_frame]["faces_not_found"] == 1:
                _means["best"].append([0, 0, 0])
                _stds["best"].append([0, 0, 0])
                _eyelids_dists["best"].append(0)
                _diameters["best"].append(0)
                _pupil2corners["best"].append(0)
            elif _frames.loc[_frame]['yaw'] > 0:
                _means["best"].append(_frames.loc[_frame]["right_mean_color"])
                _stds["best"].append(_frames.loc[_frame]["right_std_color"])
                _eyelids_dists["best"].append(
                    _frames.loc[_frame]["right_eyelids_dist"])
                _diameters["best"].append(
                    _frames.loc[_frame]["right_diameter"])
                _pupil2corners["best"].append(
                    _frames.loc[_frame]["right_pupil2corner"])
            else:
                _means["best"].append(_frames.loc[_frame]["left_mean_color"])
                _stds["best"].append(_frames.loc[_frame]["left_std_color"])
                _eyelids_dists["best"].append(
                    _frames.loc[_frame]["left_eyelids_dist"])
                _diameters["best"].append(_frames.loc[_frame]["left_diameter"])
                _pupil2corners["best"].append(
                    _frames.loc[_frame]["left_pupil2corner"])

        # save signals
        with open(means_file_path, "wb") as pkl_file:
            pickle.dump(_means, pkl_file)
        with open(stds_file_path, "wb") as pkl_file:
            pickle.dump(_stds, pkl_file)
        with open(eyelids_file_path, "wb") as pkl_file:
            pickle.dump(_eyelids_dists, pkl_file)

        with open(iris_file_path, "wb") as pkl_file:
            pickle.dump(_diameters, pkl_file)
        with open(pupil2corner_file_path, "wb") as pkl_file:
            pickle.dump(_pupil2corners, pkl_file)

        with open(yaws_file_path, "wb") as pkl_file:
            pickle.dump(_yaws, pkl_file)
        with open(pitch_file_path, "wb") as pkl_file:
            pickle.dump(_pitchs, pkl_file)
        with open(faces_not_found_file_path, "wb") as pkl_file:
            pickle.dump(_faces_not_found, pkl_file)

        print("signals are saved")

        if generate_plots:
            # generate plots
            max_mean = find_max(_means[selected_eye])
            max_std = find_max(_stds[selected_eye])
            max_dist = find_max(_eyelids_dists[selected_eye])

            max_iris = find_max(_diameters[selected_eye])
            max_pupil = find_max(_pupil2corners[selected_eye])

            for idx, _frame in tqdm.tqdm(enumerate(ordered_frames), total=len(ordered_frames)):
                _start = int(idx/_rng) * _rng
                _end = len(ordered_frames) if _start+_rng + \
                    1 > len(ordered_frames) else _start+_rng

                generate_mean_std_plot(
                    img_path=_paths[idx],
                    stds=_stds[selected_eye][_start:_end],
                    means=_means[selected_eye][_start:_end],
                    eyelids_dists=_eyelids_dists[selected_eye][_start:_end],
                    iris_diameters=_diameters[selected_eye][_start:_end],
                    faces_not_found=_faces_not_found[_start:_end],
                    bboxes=_bboxes[idx],
                    x_range=range(int(ordered_frames[_start]), int(
                        ordered_frames[_end-1])+1),
                    frame_num=int(_frame),
                    maxes=[max_mean, max_std, max_dist, max_iris],
                    mins=[0, 0, 0, 0],
                    out_folder_path=output_path,)
        else:
            print("No plots are generated, your work is done")
