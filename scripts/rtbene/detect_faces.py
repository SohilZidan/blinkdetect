#!/usr/bin/env python3
# coding: utf-8


import os
import argparse
import pandas as pd
import tqdm
import cv2
from typing import Union, Tuple
import math
from blinkdetect.common import read_annotations_tag, read_bbox_rush
from blinkdetect.image.misc import cut_region

# import mediapipe as mp
# mp_face_detection = mp.solutions.face_detection
# face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

from retinaface import RetinaFace
model = RetinaFace.build_model()



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
    # parser.add_argument(
    #     "--face_annotations",
    #     default="",
    #     help="annotations file or directory"
    #     )
    # parser.add_argument(
    #     "--dataset", type=str,
    #     choices=["rush", "rtbene", "talkingface"],
    #     default="rtbene",
    #     help="minimum number of consecutive frames, and window size"
    #     )
    return parser.parse_args()


# def _normalized_to_pixel_coordinates(
#     normalized_x: float, normalized_y: float, image_width: int,
#     image_height: int) -> Union[None, Tuple[int, int]]:
#   """Converts normalized value pair to pixel coordinates."""

#   # Checks if the float value is between 0 and 1.
#   def is_valid_normalized_value(value: float) -> bool:
#     return (value > 0 or math.isclose(0, value)) and (value < 1 or
#                                                       math.isclose(1, value))

#   if not (is_valid_normalized_value(normalized_x) and
#           is_valid_normalized_value(normalized_y)):
#     # TODO: Draw coordinates even if it's outside of the image bounds.
#     return None
#   x_px = min(math.floor(normalized_x * image_width), image_width - 1)
#   y_px = min(math.floor(normalized_y * image_height), image_height - 1)
#   return x_px, y_px


# def detect_faces(img):
#     image_rows, image_cols, _ = img.shape
#     results = face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

#     if not results.detections:
#         return None

#     detection = results.detections[0]
#     # for det in results.detections:
#     relative_bounding_box = detection.location_data.relative_bounding_box
#     rect_start_point = _normalized_to_pixel_coordinates(
#         relative_bounding_box.xmin, relative_bounding_box.ymin, image_cols,
#         image_rows)
#     rect_end_point = _normalized_to_pixel_coordinates(
#         relative_bounding_box.xmin + relative_bounding_box.width,
#         relative_bounding_box.ymin + relative_bounding_box.height, image_cols,
#         image_rows)

#     left_eye_pt_relative = mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.LEFT_EYE)
#     right_eye_pt_relative = mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.RIGHT_EYE)
#     left_eye_pt = _normalized_to_pixel_coordinates(left_eye_pt_relative.x, left_eye_pt_relative.y,
#                                 image_cols, image_rows)
#     right_eye_pt = _normalized_to_pixel_coordinates(right_eye_pt_relative.x, right_eye_pt_relative.y,
#                                 image_cols, image_rows)

#     if (
#         (rect_start_point is None) or 
#         (rect_end_point is None) or
#         (left_eye_pt is None) or
#         (right_eye_pt is None)
#         ):
#         return None

#     bbox = list(rect_start_point + rect_end_point)
#     left_eye_pt = list(left_eye_pt)
#     right_eye_pt = list(right_eye_pt)

#     face_landmarks = {
#         "facial_area" : bbox,
#         "landmarks": {
#             "left_eye": left_eye_pt,
#             "right_eye": right_eye_pt
#             }
#         }

#     return face_landmarks


if __name__ == "__main__":
    args = parse()

    # read annotations
    temporal_blinking = pd.read_hdf(args.annotations, "temporal_blinking")
    temporal_blinking = temporal_blinking.sort_index()

    # initialize
    faces = []
    left_eyes = []
    right_eyes = []
    nons = 0

    # iterate over all items
    file_paths = temporal_blinking["file_path"].values
    
    for idx, file_path in enumerate(tqdm.tqdm(file_paths, total=len(file_paths))):

        img = cv2.imread(file_path, cv2.IMREAD_COLOR)

        # face_landmarks = detect_faces(img)

        # if face_landmarks is None:
        #     nons = nons + 1
        #     faces.append([])
        #     left_eyes.append([])
        #     right_eyes.append([])
        # else:
        #     faces.append(face_landmarks["facial_area"])
        #     left_eyes.append(face_landmarks["landmarks"]["left_eye"])
        #     right_eyes.append(face_landmarks["landmarks"]["right_eye"])

        resp = RetinaFace.detect_faces(img, model=model)
        if type(resp) is tuple:
            faces.append([])
            left_eyes.append([])
            right_eyes.append([])
        else:
            # get face
            face_landmarks = resp['face_1']
            #
            faces.append(face_landmarks["facial_area"])
            left_eyes.append(face_landmarks["landmarks"]["left_eye"])
            right_eyes.append(face_landmarks["landmarks"]["right_eye"])
        


    temporal_blinking["facial_area"] = faces
    temporal_blinking["left_eye"] = left_eyes
    temporal_blinking["right_eye"] = right_eyes

    print(temporal_blinking.head())
    print(f"{nons} faces are not detected")

    # create the parent directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    temporal_blinking.to_hdf(args.output, "temporal_blinking")
