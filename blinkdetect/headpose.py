#!/usr/bin/env python3
# coding: utf-8

import os
import yaml
import sys
third_party_path = os.path.join(os.path.dirname(__file__), "..", "third-party")
lib_path = os.path.join(third_party_path, "ddfa")
sys.path.append(lib_path)

# from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.pose import calc_pose
def_config_file = os.path.join(lib_path, 'configs/mb1_120x120.yml')
def_cfg = yaml.load(open(def_config_file), Loader=yaml.SafeLoader)


class HeadPoseEstimator(TDDFA):
    def __init__(self):
        super(HeadPoseEstimator, self).__init__(gpu_mode=True, **def_cfg)

    def estimate_headpose(self, img, boxes):
        param_lst, _ = self.__call__(img, boxes)

        param = param_lst[0]
        _, pose = calc_pose(param)
        yaw = pose[0]
        pitch = pose[1]
        roll = pose[2]

        return yaw, pitch, roll
