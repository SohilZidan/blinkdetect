#!/usr/bin/env python3
# coding: utf-8

import numpy as np
from scipy.spatial.distance import directed_hausdorff

def iris_diameter(iris):
    center = iris[0, :2]
    diameter = 0.0
    diameter = np.linalg.norm(iris[1, :2] - iris[3, :2])
    diameter = np.linalg.norm(iris[2, :2] - iris[4, :2])
    return diameter/2

def eyelids_directed_hausdorff_2D(set1_indices: list, set2_indices: list, landmarks: np.ndarray):
    A = landmarks[:, set1_indices[0]:set1_indices[1], 0:2].reshape((-1,2))
    B = landmarks[:, set2_indices[0]:set2_indices[1], 0:2].reshape((-1,2))
    # diameter = iris_diameter(iris)
    return directed_hausdorff(B,A)[0]# / diameter

def eyelids_directed_hausdorff_3D(set1_indices: list, set2_indices: list, landmarks: np.ndarray):
    A = landmarks[:, set1_indices[0]:set1_indices[1], :].reshape((-1,3))
    B = landmarks[:, set2_indices[0]:set2_indices[1], :].reshape((-1,3))
    # diameter = iris_diameter(iris)
    return directed_hausdorff(B,A)[0]# / diameter

eyelids_directed_hausdorff = eyelids_directed_hausdorff_2D