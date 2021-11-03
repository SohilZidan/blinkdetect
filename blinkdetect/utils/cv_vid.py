#!/usr/bin/env python3
# coding: utf-8

import os
import shutil
from typing import List
import numpy as np
import cv2


def gen_vid(video_path: str, fps: float, width: int, height: int) -> cv2.VideoWriter:
    """create opencv videowriter with the provided properties

    Args:
        video_path (str): path to the output video
        fps (float): fps
        width (int): width
        height (int): height

    Returns:
        cv2.VideoWriter: opencv video writer
    """
    ext = video_path.split('.')[-1]
    if ext == 'mp4':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    elif ext == 'avi':
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')  #*'XVID')
    else:
        # if not .mp4 or avi, we force it to mp4
        video_path = video_path.replace(ext, 'mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case

    out = cv2.VideoWriter(video_path, fourcc, fps, (int(width), int(height)))

    return out


def gen_video(video_path: str, out_dir: str, tag: str='new') -> cv2.VideoWriter:
    """copy a video properties to another directory

    Args:
        video_path (str): path of the source video
        out_dir (str): output directory
        tag (str, optional): prefix of the new video. Defaults to 'new'.

    Returns:
        cv2.VideoWriter: opencv video writer
    """
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    #
    vid_name = os.path.basename(video_path)
    out_path = os.path.join(out_dir, tag + '_' + vid_name)
    print('Generating video: {}'.format(out_path))
    # Output folder
    if not os.path.exists(os.path.dirname(out_dir)):
        os.makedirs(os.path.dirname(out_dir))

    return gen_vid(out_path, fps, width, height)


def extract_segments(video_path: str, output_dir, segments: List=[]):
    """[summary]

    Args:
        video (str): directory path to the frames
        output_dir ([type]): [description]
        segments (List, optional): [description]. Defaults to [].
    """
    assert isinstance(video_path, str) and os.path.exists(video_path), f"{video_path} does not exist"
    assert isinstance(segments, list), "only list of indices is supported"
    assert len(segments) % 2 == 0, "segments must contain an even number of elements"

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    for start_idx in range(0, len(segments), 2):
        stop_idx = start_idx + 1

        start = segments[start_idx]
        stop = segments[stop_idx]

        frames_paths = [os.path.join(video_path, f"{i:06d}.png") for i in range(start, stop+1)]
        output_folder_name = f"{start}_{stop}"
        # new dir
        new_dir = os.path.join(output_dir, output_folder_name)
        os.makedirs(new_dir, exist_ok=True)

        for frame_src_path in frames_paths:
            frame_name = os.path.basename(frame_src_path)
            frame_dst_path = os.path.join(new_dir, frame_name)
            # shutil.copyfile(frame_src_path, frame_dst_path)
            os.symlink(frame_src_path, frame_dst_path)
