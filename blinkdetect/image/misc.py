#!/usr/bin/env python3
# coding: utf-8

import os
from typing import List, Union
import numpy as np
import cv2


def cut_region(img_path: Union[np.ndarray, str], bbox: List) -> np.ndarray:
    """
    cut a region from an image using a bounding box
    """
    assert isinstance(img_path, str) or isinstance(img_path, np.ndarray), "not a supported type"
    if isinstance(img_path, str):
        success = os.path.exists(img_path)
        if not success:
            return success, np.array([])
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    else:  # ndarray
        success = (img_path.size > 0)
        if not success:
            return False, np.array([])
        img = img_path.copy()

    # cut the face
    facial_img = img[bbox[1]:bbox[3], bbox[0]: bbox[2]]
    success = (facial_img.size > 0)

    return success, facial_img

def expand_region(bbox_org: List, img: np.ndarray, expansion_ratio: float=0.25):
        """Expand bounding box by 25% of its size

        Args:
            bbox_org (List): [description]
            img (np.ndarray): [description]

        Returns:
            [type]: [description]
        """
        H = bbox_org[3] - bbox_org[1]
        W = bbox_org[2] - bbox_org[0]

        up = int(bbox_org[1] - expansion_ratio * H)
        down = int(bbox_org[3] + expansion_ratio * H)
        left = int(bbox_org[0] - expansion_ratio * W)
        right = int(bbox_org[2] + expansion_ratio * W)

        up = up if up > 0 else 0
        down = down if down < img.shape[0] else img.shape[0]
        left = left if left > 0 else 0
        right = right if right < img.shape[1] else img.shape[1]
        bbox_m = [left, up, right, down]

        return bbox_m