#!/usr/bin/env python3
# coding: utf-8


import os
import shutil
import argparse
import numpy as np
import pandas as pd
import cv2
import tqdm

import sys
lib_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(lib_dir)

from retinaface.commons import postprocess

from blinkdetect.eyelandmarks import IrisHandler
from blinkdetect.core import points_in_between
from blinkdetect.image.misc import cut_region, expand_region
from blinkdetect.image.analyze import color_analysis
from blinkdetect.metrics.distance import iris_diameter, eyelids_directed_hausdorff

# print("PyTorch version:", torch.__version__)
# print("CUDA version:", torch.version.cuda)
# print("cuDNN version:", torch.backends.cudnn.version())
# print("OpenCV version:", cv2.__version__)

dataset_root = os.path.join(os.path.dirname(__file__), "..", "dataset")


def parser():
    _parser = argparse.ArgumentParser()
    _parser.add_argument(
        '--dataset', default="BlinkingValidationSetVideos",
        choices=["BlinkingValidationSetVideos", "eyeblink8", "talkingFace", "zju", "RN"])
    # _parser.add_argument('-rng', '--range', type=int, default=[0,-1], nargs=2)
    # _parser.add_argument('--batch', type=int, default=32, help='number of frames to be saved as a batch')
    # _parser.add_argument('--resume', action='store_true', help='if true existed frames of an existed participant will not be replaced')
    # _parser.add_argument('--save_faces', type=str, default="", help='wheather to save faces or not')
    # _parser.add_argument('--dim', required=True, choices=["3D", "2D"], help="decide in which dimension to compute eyelids distance")

    return _parser.parse_args()


def overlay_image(aligned_facial_img: np.ndarray, resp):
    aligned_facial_img_eyelids = aligned_facial_img.copy()
    aligned_facial_img_iris = aligned_facial_img.copy()

    for eye_key in resp:
        org_eye, org_iris = resp[eye_key]    
        # eyelids
        for i in range(1, 16):
            center = org_eye[0, i, 0:2].astype(np.int)
            aligned_facial_img_eyelids = cv2.circle(aligned_facial_img_eyelids, (center[0], center[1]), radius=0, color=(255, 255, 100), thickness=2)
        # iris
        for i in range(5):
            center = org_iris[i,0:2].astype(np.int)
            aligned_facial_img_iris = cv2.circle(aligned_facial_img_iris, (center[0], center[1]), radius=0, color=(255, 0, 0), thickness=2)

        # draw pupil circle
        h_diameter = np.linalg.norm(org_iris[1,:] - org_iris[3,:])
        v_diameter = np.linalg.norm(org_iris[2,:] - org_iris[4,:])
        # h_diameters.append(h_diameter)
        # v_diameters.append(v_diameter)
        pupil_center = org_iris[0,0:2].astype(np.int)
        aligned_facial_img_iris = cv2.circle(aligned_facial_img_iris, (pupil_center[0], pupil_center[1]), radius=round(min(h_diameter, v_diameter)/2), color=(0, 0, 255), thickness=1)

    aligned_facial_img = np.hstack([aligned_facial_img, aligned_facial_img_iris, aligned_facial_img_eyelids])
    return aligned_facial_img


def extract_eye_features(img_path: str, face_landmarks, iris_marker):
    """"""

    # read image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # margin
    # new
    bbox_m = expand_region(face_landmarks["facial_area"], img)

    # crop face
    _, facial_img = cut_region(img, bbox_m)

    # horizontally alignment 
    aligned_facial_img = postprocess.alignment_procedure(
        facial_img,
        face_landmarks["landmarks"]["right_eye"], face_landmarks["landmarks"]["left_eye"])

    # extract eye landmarks
    resp = iris_marker.extract_eye_landmarks(aligned_facial_img)

    # draw
    
    # org_aligned_img = cv2.cvtColor(aligned_facial_img, cv2.COLOR_RGB2BGR)
    # tmp_image = overlay_image(org_aligned_img, resp)
    # tmp_file_out = f"{os.path.sep}".join(img_path.split(os.path.sep)[-3:])
    # tmp_file_out = os.path.join("/home/zidan/blinkdetection/tmp_dataset", tmp_file_out)
    # os.makedirs(os.path.dirname(tmp_file_out), exist_ok=True)
    # cv2.imwrite(tmp_file_out, tmp_image)

    # extract features
    eyelidsDistances = dict.fromkeys(resp.keys())
    irisesDiameters = dict.fromkeys(resp.keys())
    pupil2corners = dict.fromkeys(resp.keys())
    meanColorCurve = dict.fromkeys(resp.keys())
    stdColorCurve = dict.fromkeys(resp.keys())


    for _eye in resp:
        org_eye, org_iris = resp[_eye]

        # eyelids distance
        eyelidsDistances[_eye] = eyelids_directed_hausdorff(set1_indices=[1,8], set2_indices=[9,16], landmarks=org_eye)
        # iris diameter
        irisesDiameters[_eye] = iris_diameter(org_iris)
        # distance from puipl center to the left corner (inner corner)
        pupil2corners[_eye] = np.linalg.norm(org_iris[0,0:2] - org_eye[0, 0, 0:2])

        mid_points = points_in_between(org_eye)
        _std, _mean = color_analysis(aligned_facial_img, mid_points)
        # mean and std for 3 channels
        meanColorCurve[_eye] = _std
        stdColorCurve[_eye] = _mean

    return eyelidsDistances, irisesDiameters, pupil2corners, meanColorCurve, stdColorCurve


if __name__=='__main__':
    # 
    args = parser()
    dataset = os.path.join(dataset_root, args.dataset)

    # iris handler
    iris_marker = IrisHandler()

    #
    # video paths
    videos_paths = []
    for root,dirs, files in os.walk(dataset):
        for dir in files:
            name, ext = os.path.splitext(dir)
            if ext in [".avi", ".mov", ".wmv", ".mp4"]:
                videos_paths.append(os.path.join(root,dir))
    #
    #
    videos_progress = tqdm.tqdm(videos_paths, total=len(videos_paths), desc="landmarks")
    for video_path in videos_progress:
        # input 
        video_name = os.path.dirname(video_path)
        video_name = os.path.relpath(video_name, dataset)
        videos_progress.set_postfix(video=video_name)

        frames_path=os.path.join(os.path.dirname(video_path), "frames")
        if not os.path.exists(frames_path):
            print(f"frames folder {frames_path} not found")
            continue

        input_path = os.path.join(dataset_root, 'tracked_faces', args.dataset , video_name)
        input_path_file_hdf5 = os.path.normpath(os.path.join(input_path, "faceinfo.hdf5"))

        if not os.path.exists(input_path_file_hdf5):
            print(f"faces info file {input_path_file_hdf5} not found")
            continue
    
        # read input
        with pd.HDFStore(input_path_file_hdf5) as store:
            _data_df = store['tracked_faces_dataset_01']

        # Output
        output_path = os.path.join(dataset_root, "eye_landmarks_v2", args.dataset , video_name)
        output_file_path_csv = os.path.normpath(os.path.join(output_path, f"eyeinfo.csv"))
        output_file_path_hdf5 = os.path.normpath(os.path.join(output_path, f"eyeinfo.hdf5"))

        shutil.rmtree(output_path, ignore_errors=True)
        os.makedirs(output_path, exist_ok=True)

        # get subject data
        # face_id = _data_df.groupby('face_id').count().idxmax()[0]
        # _data_df = _data_df.loc[(_data_df.index.get_level_values('participant_id') == video_name) & (_data_df.index.get_level_values('face_id') == face_id)].reset_index(level=['participant_id', 'face_id'])
        # initialize
        eyelids_dists = {"left": [], "right": []}
        iris_diameters = {"left": [], "right": []}
        pupil_dists = {"left": [], "right": []}
        colors_means = {"left": [], "right": []}
        colors_stds = {"left": [], "right": []}

        for index, row in tqdm.tqdm(_data_df.iterrows(), total=_data_df.shape[0], desc="frame"):
            faces_not_found = int(row['faces_not_found'])
            # if len(facial_area) == 0:
            if faces_not_found:
                eyelidsDistances = {"left": -1, "right": -1}
                irisesDiameters = {"left": -1, "right": -1}
                pupil2corners = {"left": -1, "right": -1}
                meanColorCurve = {"left": [], "right": []}
                stdColorCurve = {"left": [], "right": []}
            else:
                facial_area = row[['left', 'top', 'right', 'bottom']].astype(np.float)
                left_eye = row[['left_eye_x', 'left_eye_y']].astype(np.float)
                right_eye = row[['right_eye_x', 'right_eye_y']].astype(np.float)

                face_landmarks = {
                    "facial_area" : facial_area,
                    "landmarks": {
                        "left_eye": left_eye,
                        "right_eye": right_eye
                        }
                    }

                img_path = row["img_path"]

                eyelidsDistances, irisesDiameters, pupil2corners, meanColorCurve, stdColorCurve = extract_eye_features(img_path, face_landmarks, iris_marker)
                


            # append the new data
            for eye_idx in eyelidsDistances.keys():
                eyelids_dists[eye_idx].append(eyelidsDistances[eye_idx])
                iris_diameters[eye_idx].append(irisesDiameters[eye_idx])
                pupil_dists[eye_idx].append(pupil2corners[eye_idx])
                colors_means[eye_idx].append(meanColorCurve[eye_idx])
                colors_stds[eye_idx].append(stdColorCurve[eye_idx])

        for eye_key in eyelids_dists.keys():
            _data_df[f"{eye_key}_eyelids_dist"] = eyelids_dists[eye_key]
            _data_df[f"{eye_key}_diameter"] = iris_diameters[eye_key]
            _data_df[f"{eye_key}_pupil2corner"] = pupil_dists[eye_key]
            _data_df[f"{eye_key}_mean_color"] = colors_means[eye_key]
            _data_df[f"{eye_key}_std_color"] = colors_stds[eye_key]

        # save
        _data_df.to_hdf(output_file_path_hdf5, "eyes_info_dataset_01")
        print(_data_df.head())

    videos_progress.close()