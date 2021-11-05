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