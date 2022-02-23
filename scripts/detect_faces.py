#!/usr/bin/env python3
# coding: utf-8

import os
import glob
import argparse
from math import ceil
import pickle
import tqdm
from blinkdetect.core.face import extract_face
from configs import load_cfgs


def extract_faces(
        images_paths: list,
        output_dir: str = './output_faces'):

    os.makedirs(output_dir, exist_ok=True)

    detections = {}

    for _img_path in tqdm.tqdm(images_paths, total=len(images_paths), leave=False, desc="frame"):
        img_name = os.path.basename(_img_path)
        _name, _ = img_name.split(".")

        # face detection
        dets = extract_face(_img_path)

        detections[_name] = dets

    return detections


def parser(base_cfgs):
    _parser = argparse.ArgumentParser()
    _parser.add_argument(
        '--dataset', required=True,
        choices=base_cfgs.get("DATASETS"))
    _parser.add_argument(
        '--batch', type=int, default=256,
        help='number of frames to be saved as a batch')
    _parser.add_argument(
        '--resume', action='store_true',
        help="""if true, 
        existed frames of an existed participant will not be replaced""")
    return _parser.parse_args()


dataset_root = os.path.join(os.path.dirname(__file__), "..", "dataset")


if __name__ == "__main__":
    base_cfgs = load_cfgs()
    args = parser(base_cfgs)
    resume = args.resume
    dataset = os.path.normpath(os.path.join(dataset_root, args.dataset))

    # video paths
    videos_paths = []
    for root, dirs, files in os.walk(dataset):
        for dir in files:
            name, ext = os.path.splitext(dir)
            if ext in [".avi", ".mov", ".wmv", ".mp4"]:
                videos_paths.append(os.path.join(root, dir))

    videos_progress = tqdm.tqdm(videos_paths, total=len(
        videos_paths), desc="face detection")

    for video_path in videos_progress:
        video_name = os.path.dirname(video_path)
        video_name = os.path.relpath(video_name, dataset)

        videos_progress.set_postfix(video=video_name)

        # input
        frames_root = os.path.normpath(os.path.join(
            os.path.dirname(video_path), "frames"))

        if not os.path.exists(frames_root):
            continue

        # output
        faces_root = os.path.normpath(os.path.join(
            dataset_root, "faces", args.dataset, video_name))
        faceinfo_file_path_pkl = os.path.normpath(
            os.path.join(faces_root, "faceinfo.pkl"))

        os.makedirs(faces_root, exist_ok=True)

        # when resume is set, existed participant_id,frame_num indices will not be processed
        _except_frames = []
        _all_detections = {}
        if resume:
            if os.path.exists(faceinfo_file_path_pkl) and os.path.getsize(faceinfo_file_path_pkl) > 0:
                with open(faceinfo_file_path_pkl, "rb") as pkl_file:
                    _all_detections = pickle.load(pkl_file)
                    _except_frames.extend(list(_all_detections.keys()))

        # load images
        frames_excepted = set(_except_frames)
        _images = set(glob.glob(f"{frames_root}/*.png"))
        _images = sorted(_images.difference(frames_excepted))
        total_imgs = len(_images)
        batch = args.batch

        iteration_progress = tqdm.tqdm(
            range(0, total_imgs, batch), total=total_imgs, leave=False, desc="batch")
        for i in iteration_progress:

            _new_detections = extract_faces(
                images_paths=_images[i: i + batch],
                output_dir=faces_root)

            _all_detections = {**_all_detections, **_new_detections}

            # save the results
            with open(faceinfo_file_path_pkl, "wb") as pkl_file:
                pickle.dump(_all_detections, pkl_file)

            iteration_progress.set_postfix(
                saved=f"{len(_all_detections)} dets saved")
        iteration_progress.close()

    videos_progress.close()
