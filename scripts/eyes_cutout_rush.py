#!/usr/bin/env python3

import os
import argparse
import shutil
import pickle
import tqdm
import cv2
import torch
import numpy as np
from retinaface.commons.postprocess import alignment_procedure
from bdlib.models.facemesh import FaceMesh
from bdlib.models.irislandmarks import IrisLandmarks

print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())
print("OpenCV version:", cv2.__version__)
gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
FACE_MESH_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "facemesh.pth")
IRIS_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "irislandmarks.pth")

net = FaceMesh().to(gpu)
net.load_weights(FACE_MESH_MODEL_PATH)
iris_net = IrisLandmarks().to(gpu)
iris_net.load_weights(IRIS_MODEL_PATH)

lower_right = [33, 7, 163, 144, 145, 153, 154, 155]
upper_right = [133, 173, 157, 158, 159, 160, 161, 246]
lower_left = [362, 382, 381, 380, 374, 373, 390, 249]
upper_left = [263, 466, 388, 387, 386, 385, 384, 398]

dataset_root = os.path.join(os.path.dirname(__file__), "..", "dataset")


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", required=True,
        choices=["BlinkingValidationSetVideos", "RN", "talkingFace"])
    return parser.parse_args()


if __name__ == "__main__":
    args = args_parser()
    dataset = os.path.normpath(os.path.join(dataset_root, args.dataset))
    #
    # video paths
    #
    videos_paths = []
    for root, dirs, files in os.walk(dataset):
        for dir in files:
            name, ext = os.path.splitext(dir)
            desired_tags = [".mp4", ".avi"]
            if ext in desired_tags:
                videos_paths.append(os.path.join(root, dir))

    videos_progress = tqdm.tqdm(
        videos_paths, total=len(videos_paths),
        desc="eye-cutout")
    for video_path in videos_progress:
        # input
        video_name = os.path.dirname(video_path)
        video_name = os.path.relpath(video_name, dataset)
        videos_progress.set_postfix(video=video_name)
        frames_root = os.path.normpath(
            os.path.join(os.path.dirname(video_path), "frames")
        )
        faces_detection_file_path = os.path.normpath(
            os.path.join(
                dataset_root, "faces", args.dataset,
                video_name, 'faceinfo_v2.pkl')
        )
        # load detections
        with open(faces_detection_file_path, "rb") as _dets_file:
            faces_detections = pickle.load(_dets_file)
        # output
        new_dir = os.path.join(
            dataset_root, "eye-cutouts", args.dataset, video_name
        )

        #
        if os.path.exists(new_dir):
            shutil.rmtree(new_dir)
        os.makedirs(new_dir)
        h_diameters = []
        v_diameters = []
        frames = sorted(os.listdir(frames_root))
        for frame in tqdm.tqdm(frames, total=len(frames)):
            name, ext = os.path.splitext(frame)
            frame_path = os.path.join(frames_root, frame)
            # face detection
            img = cv2.imread(frame_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ###############
            if faces_detections[name]['faces_not_found']:
                continue
            faces = faces_detections[name]['faces']
            ###############
            #
            bbox_org = faces['face_1']['facial_area']
            left_eye = faces['face_1']['landmarks']['left_eye']
            right_eye = faces['face_1']['landmarks']['right_eye']
            nose = faces['face_1']['landmarks']['nose']
            facial_img_org = (
                img[bbox_org[1]:bbox_org[3], bbox_org[0]:bbox_org[2]]
            )
            # 0.25 margin
            H = bbox_org[3] - bbox_org[1]
            W = bbox_org[2] - bbox_org[0]
            up = int(bbox_org[1]-0.25*H) if int(bbox_org[1]-0.25*H) > 0 else 0
            down = (
                int(bbox_org[3]+0.25*H)
                if int(bbox_org[3]+0.25*H) < img.shape[0] else img.shape[0]
            )
            left = (
                int(bbox_org[0]-0.25*W) if int(bbox_org[0]-0.25*W) > 0 else 0
            )
            right = (
                int(bbox_org[2]+0.25*W)
                if int(bbox_org[2]+0.25*W) < img.shape[1] else img.shape[1]
            )
            bbox_m = [left, up, right, down]
            facial_img_m = img[bbox_m[1]:bbox_m[3], bbox_m[0]:bbox_m[2]]
            # alignment
            aligned_facial_img = alignment_procedure(
                facial_img_m, right_eye, left_eye
            )
            # face mesh
            img_mesh = aligned_facial_img.copy()
            img_mesh = cv2.resize(img_mesh, (192, 192))
            face_resized = img_mesh.copy()
            detections = net.predict_on_image(img_mesh).cpu().numpy()
            #
            left_eye = lower_left+upper_left
            right_eye = lower_right+upper_right
            _detections = {"left": None, "right": None}
            detections_left = np.array(
                list(map(detections.__getitem__, left_eye))
            )
            detections_right = np.array(
                list(map(detections.__getitem__, right_eye))
            )
            _detections["left"] = detections_left
            _detections["right"] = detections_right
            for eye_key in _detections.keys():
                x, y = _detections[eye_key][:, 0], _detections[eye_key][:, 1]
                # dets = detections.astype(np.int32)
                height, width = 192, 192
                eye_center = []
                eye_center.append(np.mean(x, axis=0))
                eye_center.append(np.mean(y, axis=0))
                x1 = eye_center[0]
                y1 = eye_center[1]
                _enlarge = 40
                _enlarge = min([x1, width-x1, y1, height-y1, _enlarge])
                left = int(x1-_enlarge)
                right = int(x1+_enlarge)
                top = int(y1-_enlarge/2)
                bottom = int(y1+_enlarge/2)
                eye_region_org = img_mesh[top:bottom, left:right, :]
                # iris-landmarks
                eye_region = cv2.resize(eye_region_org, (64, 64))
                eye_gpu, iris_gpu = iris_net.predict_on_image(eye_region)
                eye = eye_gpu.cpu().numpy()
                # iris = iris_gpu.cpu().numpy()
                min_coord = np.min(eye.reshape(-1, 3)[:, :2], axis=0)
                max_coord = np.max(eye.reshape(-1, 3)[:, :2], axis=0)
                X, Y = min_coord.astype(np.int)
                X1, Y1 = max_coord.astype(np.int)
                eye_region = eye_region[Y:Y1, X:X1, :]
                #
                eye_region = cv2.cvtColor(eye_region, cv2.COLOR_RGB2BGR)
                eye_path = os.path.join(new_dir, eye_key)
                if not os.path.exists(eye_path):
                    os.makedirs(eye_path)
                frame_path = os.path.join(eye_path, frame)
                cv2.imwrite(frame_path, eye_region)
