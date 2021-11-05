#!/usr/bin/env python3
# coding: utf-8

from typing import Tuple, List
import numpy as np
import scipy.ndimage


def color_analysis(img_region: np.ndarray, points: np.ndarray) -> Tuple[List, List]:
    """
    compute the color values std and mean along a curve of sampled `points`

    Args:
        img_region (np.ndarray): 3 Channels image
        points (np.ndarray): shape of (N, 2)

    Returns:
        Tuple[List, List]: a tuple of 2 elements. The first is a list of stds,
        whereas the second is the means
    """
    # mid_points = np.stack([midx,midy], axis=1)
    diff_mid_points = points[1:] - points[:-1]
    _nums = np.ceil(np.linalg.norm(diff_mid_points, axis=1)).astype(np.int32)

    acc_x = np.array([])
    acc_y = np.array([])
    for i in range(len(diff_mid_points)):
        num_t = _nums[i]
        acc_x = np.hstack((acc_x, np.linspace(midx[i], midx[i+1], num_t)))
        acc_y = np.hstack((acc_y, np.linspace(midy[i], midy[i+1], num_t)))

    x, y = acc_x, acc_y

    _std = []
    _mean = []
    for _dim in range(img_region.shape[2]):
        z = img_region[:,:,_dim].copy()
        zi = scipy.ndimage.map_coordinates(z, np.vstack((y,x)))
        # measurements  
        _std.append(np.std(zi))
        _mean.append(np.mean(zi))

    return _std, _mean
