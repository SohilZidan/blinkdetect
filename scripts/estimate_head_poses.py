# coding: utf-8

__author__ = 'sohilzidan'

import argparse
import cv2
import yaml
import os
import glob
import pickle
from math import ceil
import tqdm

third_party_path = os.path.join(os.path.dirname(__file__), "..", "third-party")
lib_path = os.path.join(third_party_path, "ddfa")
import sys
sys.path.append(lib_path)

# from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.pose import calc_pose
from utils.functions import plot_image
from utils.pose import plot_pose_box





def_config = os.path.join(lib_path, 'configs/mb1_120x120.yml')


def main(args):
    cfg = yaml.load(open(args.config), Loader=yaml.SafeLoader)
    # participant_id = args.participant_id
    start, end = args.range
    output_file_name = args.output_file

    dataset_root = os.path.join(os.path.dirname(__file__), "..", "dataset")
    dataset = os.path.normpath(os.path.join(dataset_root, args.dataset))

    # videos paths
    videos_paths = []
    for root,dirs, files in os.walk(dataset):
        for dir in files:
            name, ext = os.path.splitext(dir)
            if ext in [".avi", ".mov", ".wmv", ".mp4"]:
                videos_paths.append(os.path.join(root,dir))
    #
    # 
    for video_path in videos_paths:
        # input 
        video_name = os.path.dirname(video_path)
        video_name = os.path.relpath(video_name, dataset)
        frames_root=os.path.join(os.path.dirname(video_path), "frames")
        if not os.path.exists(frames_root):
            continue
        # 
        faces_detection_file_path = os.path.normpath(os.path.join(dataset_root, "faces", args.dataset, video_name, 'faceinfo.pkl'))
        
        # output
        faces_detection_with_pose_file_path = os.path.normpath(os.path.join(dataset_root,"faces", args.dataset, video_name, output_file_name))

        if not os.path.exists(faces_detection_file_path):
            print(f"faces detection file {faces_detection_file_path} not found")
            continue

        #
        gpu_mode = args.mode == 'gpu'
        tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
        # face_boxes = FaceBoxes()

        # load images
        images_paths = sorted(glob.glob(f"{frames_root}/*.png")) 
        if end == -1: end = len(glob.glob(f"{frames_root}/*.png"))

        # load detections
        with open(faces_detection_file_path, "rb") as _dets_file:
            _detections = pickle.load(_dets_file)

        total_range = end-start
        batch = args.batch
        _iterations = ceil(total_range/batch)

        for i in tqdm.tqdm(range(_iterations), total=_iterations):
            _batch_start = start+batch*i
            _batch_end = min(start+batch*(i+1),end)
            # 
            _images = images_paths[_batch_start:_batch_end]
            # 
            for _img_idx in tqdm.tqdm(range(len(_images)), total=len(_images)):
                # 
                _img_path = _images[_img_idx]
                img_name = os.path.basename(_img_path)
                _name, _ext = img_name.split(".")

                # Given a still image path and load to BGR channel
                img = cv2.imread(_img_path)

                # 
                if type(_detections[_name]['faces']) is not dict:
                    print(f'No face detected')
                    continue
                faces = _detections[_name]['faces']
                faces_ids = sorted(faces.keys())
                boxes = [faces[face]['facial_area'] for face in faces_ids]

                # 
                param_lst, roi_box_lst = tddfa(img, boxes)

                dense_flag = False
                ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)

                for _idx, (param, ver) in enumerate(zip(param_lst, ver_lst)):
                    P, pose = calc_pose(param)
                    faces[faces_ids[_idx]]['yaw'] = pose[0]
                    faces[faces_ids[_idx]]['pitch'] = pose[1]
                    faces[faces_ids[_idx]]['roll'] = pose[2]

                    if (_img_idx+1) % batch == 0 and args.show_sample:
                        img = plot_pose_box(img, P, ver)
                        print(f'yaw: {pose[0]:.1f}, pitch: {pose[1]:.1f}, roll: {pose[2]:.1f}')
                        plot_image(img)


            # _all_detections = {**}
            with open( faces_detection_with_pose_file_path, "wb" ) as pkl_file:
                pickle.dump(_detections, pkl_file)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The demo of still image of 3DDFA_V2')
    parser.add_argument('--dataset', required=True, choices=["BlinkingValidationSetVideos", "eyeblink8", "talkingFace", "zju", "RN"])
    parser.add_argument('-rng', '--range', type=int, default=[0,-1], nargs=2)
    parser.add_argument('--batch', type=int, default=32, help='number of frames to be saved as a batch')
    parser.add_argument('--resume', action='store_true', help='if true existed frames of an existed participant will not be replaced')
    # 
    parser.add_argument('-c', '--config', type=str, default=def_config)
    parser.add_argument('-m', '--mode', type=str, default='cpu', help='gpu or cpu mode')
    parser.add_argument('--output_file', type=str, default='faceinfo_v2.pkl')
    parser.add_argument("--show_sample", action="store_true")

    args = parser.parse_args()
    main(args)
