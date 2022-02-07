#!/usr/bin/env python3
# coding: utf-8

import os
from typing import Dict, Any, Union
import pickle
import tqdm
import numpy as np
from retinaface import RetinaFace

retina_detect_faces = RetinaFace.detect_faces
model = RetinaFace.build_model()


def extract_face(img_path: Union[str, np.ndarray]) -> Dict[str, Any]:
    """predicts faces bboxes using retina-face

    Args:
        img_path Union[str, np.ndarray]: image path

    Returns:
        Dict[str, Any]: keys(faces, faces_number, faces_not_found).
        faces is None when no face is found
    """

    # face detection
    dets = retina_detect_faces(img_path=img_path, model=model)

    if type(dets) is tuple:
        faces_not_found = 1
        n_faces = 0
    else:
        faces_not_found = 0
        n_faces = len(dets.keys())

    detections = {
        "faces": dets,
        "faces_not_found": faces_not_found,
        "faces_number": n_faces
    }

    return detections
