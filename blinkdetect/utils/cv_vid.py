#!/usr/bin/env python3
# coding: utf-8

import cv2
import numpy as np
import os

def gen_vid(video_path, fps, width, height):
    

    # Combine video
    ext = video_path.split('.')[-1]
    if ext == 'mp4':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    elif ext == 'avi':
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')  #*'XVID')
    else:
        # if not .mp4 or avi, we force it to mp4
        video_path = video_path.replace(ext, 'mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case


    out = cv2.VideoWriter(video_path, fourcc, fps, (np.int32(width), np.int32(height)))

    return out

def gen_video(video_path, out_dir, tag='3ddf'):
    # 
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