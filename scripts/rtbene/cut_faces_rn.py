#!/usr/bin/env python3
# coding: utf-8


import os
import argparse
import pandas as pd
from pathlib import Path
import tqdm
import cv2
from blinkdetect.common import read_bbox_rush, read_bbox_tag
from blinkdetect.image.misc import cut_region




def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations",
        required=True,
        help="annotations file"
        )
    parser.add_argument(
        "--output",
        required=True,
        help="output folder"
        )
    parser.add_argument(
        "--part",
        default="test/rn30",
        help="part of the dataset to handle"
        )

    return parser.parse_args()


def expand_region(bbox_org, img, expansion_ratio=0.25):
        """Expand bounding box by expansion_ratio of its size

        Args:
            bbox_org (List): [description]
            img (np.ndarray): [description]

        Returns:
            [type]: [description]
        """
        # H = bbox_org[3] - bbox_org[1]
        # W = bbox_org[2] - bbox_org[0]
        H = bbox_org[3]
        W = bbox_org[2]
        bbox_tmp = [bbox_org[0], bbox_org[1], bbox_org[0]+W, bbox_org[1]+H]

        up = int(bbox_tmp[1] - expansion_ratio * H)
        down = int(bbox_tmp[3] + expansion_ratio * H)
        left = int(bbox_tmp[0] - expansion_ratio * W)
        right = int(bbox_tmp[2] + expansion_ratio * W)

        up = up if up > 0 else 0
        down = down if down < img.shape[0] else img.shape[0]
        left = left if left > 0 else 0
        right = right if right < img.shape[1] else img.shape[1]
        bbox_m = [left, up, right, down]

        return bbox_m


def ensure_path(path, dirpath = False):
    """ensure subsequent/nested dir exists"""
    dir = path
    if not dirpath:
        dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)


if __name__ == "__main__":
    args = parse()

    # read annotations
    temporal_blinking = pd.read_hdf(args.annotations, "temporal_blinking")
    temporal_blinking = temporal_blinking.sort_index()

    face_annotations_paths = list(Path("/home/zidan/blinkdetection/dataset/RN/test/rn30").glob("**/*.tag"))
    _subjects = [_path.parent.parts[-1] for _path in face_annotations_paths]
    # print(Path(f"/home/zidan/blinkdetection/dataset/RN/{args.part}"))
    # print(list(Path("/home/zidan/blinkdetection/dataset/RN/test/rn30").glob("**/*.tag")))
    # print(list(face_annotations_paths))
    # print(_subjects)
    # exit()
    #
    faceinfo = dict.fromkeys(_subjects)
    for idx, _file in enumerate(face_annotations_paths):
        r_s = _subjects[idx]
        faceinfo[r_s] = read_bbox_tag(_file, mode="XYXY")

    # iterate over all items
    file_paths = temporal_blinking["file_path"].values
    # dataset_path = os.path.basename(args.face_annotations)

    progress_bar = tqdm.tqdm(file_paths, total=len(file_paths))
    for idx, file_path in enumerate(progress_bar):

        img = cv2.imread(file_path, cv2.IMREAD_COLOR)

        _index = temporal_blinking.iloc[idx].name
        _frame = temporal_blinking.iloc[idx]["frame"]
        _frame = f"{int(_frame):06d}"
        r_s = _index[0]
        # try:
        bbox_org = faceinfo[r_s][_frame]
        # except:
            # print(r_s)
            # print(_frame)
            # print(faceinfo[r_s])
            # print(faceinfo[r_s][_frame])
        bbox = expand_region(bbox_org, img)
        _, img = cut_region(img, bbox)
        
        
        t_seq = file_path.split(os.sep)
        t_seq[-6] = "rn_faces"
        new_path = f"{os.sep}".join(t_seq)
        
        ensure_path(new_path)
        if img.size == 0:
            continue
        cv2.imwrite(new_path, img)

        temporal_blinking.iloc[idx, 1] = new_path
        progress_bar.set_postfix(frame=os.path.basename(temporal_blinking.iloc[idx, 1]))

    print(temporal_blinking.head())
    print(f"{temporal_blinking.shape}")

    # create the parent directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    temporal_blinking.to_hdf(args.output, "temporal_blinking")
