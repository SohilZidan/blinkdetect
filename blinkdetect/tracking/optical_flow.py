#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import cv2


def compute_optFlow(prev_frame: np.ndarray, current_frame: np.ndarray):
    """
    Computes a dense optical flow using the Gunnar Farneback's algorithm.
    """
    mask = np.zeros_like(current_frame)
    # Sets image saturation to maximum
    mask[..., 1] = 255
    if prev_frame is None:
        return mask
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    # Calculates dense optical flow by Farneback method
    flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, 
                                    None,
                                    0.5, 3, 11, 3, 5, 1.2, 0)

    # Computes the magnitude and angle of the 2D vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Sets image hue according to the optical flow 
    # direction
    mask[..., 0] = angle * 180 / np.pi / 2
    
    # Sets image value according to the optical flow
    # magnitude (normalized)
    mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

    # Converts HSV to RGB (BGR) color representation
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
    
    # Opens a new window and displays the output frame
    return rgb
