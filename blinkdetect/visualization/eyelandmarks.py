import argparse
import os
import shutil
import cv2
import torch
import numpy as np
from retinaface import RetinaFace
from retinaface.commons.postprocess import alignment_procedure
import matplotlib.pyplot as plt
import tqdm
from typing import List
print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())
print("OpenCV version:", cv2.__version__)
# torch.cuda.device_count()
gpu0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gpu1 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

from blinkdetect.models.facemesh import FaceMesh
from blinkdetect.models.irislandmarks import IrisLandmarks

FACE_MESH_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "facemesh.pth")
IRIS_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "irislandmarks.pth")

lower_right = [33, 7, 163, 144, 145, 153, 154, 155]
upper_right = [133, 173, 157, 158, 159, 160, 161, 246]
lower_left = [362, 382, 381, 380, 374, 373, 390, 249]
upper_left = [263, 466, 388, 387, 386, 385, 384, 398]

class IrisMarker(object):
    """
    uses media pipe to draw iris center, iris contour and eyelids landmarks
    """
    def __init__(
        self,
        facemesh_path: str = FACE_MESH_MODEL_PATH, iris_path: str = IRIS_MODEL_PATH,
        face_detector: bool=True):
        super(IrisMarker, self).__init__()

        if face_detector:
            self.model = RetinaFace.build_model()
        else:
            self.model == None

        self._expansion_ratio = 0.25

        self.facemesh_net = FaceMesh().to(gpu0)
        self.facemesh_net.load_weights(facemesh_path)

        self.iris_net = IrisLandmarks().to(gpu1)
        self.iris_net.load_weights(iris_path)
    
    def expand_bbox(self, bbox_org: List, img: np.ndarray):
        H = bbox_org[3] - bbox_org[1]
        W = bbox_org[2] - bbox_org[0]
        #
        up = int(bbox_org[1]-self._expansion_ratio*H)
        down = int(bbox_org[3]+self._expansion_ratio*H)
        left = int(bbox_org[0]-self._expansion_ratio*W)
        right = int(bbox_org[2]+self._expansion_ratio*W)
        #
        up = up if up > 0 else 0
        down = down if down < img.shape[0] else img.shape[0]
        left = left if right > 0 else 0
        right = right if right < img.shape[1] else img.shape[1]
        bbox_m = [left, up, right, down]
    
        return bbox_m
    
    def overlay_image(self, img: np.ndarray, transform = False):
        """This expects input to be RGB if transform is False.
        if transform is True, It will convert the input into RGB

        Args:
            image (np.ndarray): [description]
            transform (bool, optional): [description]. Defaults to False.
        """
        assert self.model is not None, "this version does not support other bounding boxes formats"

        if transform:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # check validity
        faces = RetinaFace.detect_faces(img, model=self.model)
        if type(faces) is tuple:
            return img
        # in case there is a face take the first one
        bbox_org = faces['face_1']['facial_area']
        left_eye = faces['face_1']['landmarks']['left_eye']
        right_eye = faces['face_1']['landmarks']['right_eye']
        # nose = faces['face_1']['landmarks']['nose']
        # facial_img_org = img[bbox_org[1]:bbox_org[3], bbox_org[0]:bbox_org[2]]
        # margin
        bbox_m = self.expand_bbox(bbox_org, img)
        # [up:down,left:right]
        # left, up, right, down
        facial_img_m = img[bbox_m[1]:bbox_m[3], bbox_m[0]:bbox_m[2]]
        # alignment
        aligned_facial_img = alignment_procedure(facial_img_m, left_eye, right_eye)
        # face mesh
        img_mesh = aligned_facial_img.copy()
        img_mesh = cv2.resize(img_mesh, (192, 192))
        # face_resized = img_mesh.copy()
        detections = self.facemesh_net.predict_on_image(img_mesh).cpu().numpy()
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
        #
        for eye_key in _detections.keys():
            x, y = _detections[eye_key][:, 0], _detections[eye_key][:, 1]
            height, width = 192, 192
            eye_center = []
            eye_center.append(np.mean(x,axis=0))
            eye_center.append(np.mean(y,axis=0))
            x1=eye_center[0]
            y1=eye_center[1]
            _enlarge = 40
            _enlarge = min([x1, width-x1, y1, height-y1,_enlarge])
            left=int(x1-_enlarge)
            right=int(x1+_enlarge)
            top=int(y1-_enlarge)
            bottom=int(y1+_enlarge)
            eye_region_org = img_mesh[top:bottom,left:right, :]
            # iris landmarks
            eye_region = eye_region_org.copy()
            eye_region = cv2.resize(eye_region, (64, 64))
            eye_gpu, iris_gpu = self.iris_net.predict_on_image(eye_region)
            eye = eye_gpu.cpu().numpy()
            iris = iris_gpu.cpu().numpy()
            # scaling back
            e_H, e_W, _ = eye_region_org.shape
            e_scale_y, e_scale_x = e_H/64, e_W/64
            f_H, f_W, _ = facial_img_m.shape
            f_scale_y, f_scale_x = f_H/192, f_W/192
            #
            org_iris = ((iris[0,:, :2] @ np.array([[e_scale_x, 0],[0,e_scale_y] ])) + np.array([left, top])) @ np.array([[f_scale_x, 0],[0,f_scale_y]])
            org_eye = ((eye[:,:,:2] @ np.array([[e_scale_x, 0],[0,e_scale_y] ])) + np.array([left, top])) @ np.array([[f_scale_x, 0],[0,f_scale_y]])
            # draw iris landmarks
            for i in range(5):
                center = org_iris[i,0:2].astype(np.int)
                aligned_facial_img = cv2.circle(aligned_facial_img, (center[0], center[1]), radius=0, color=(255, 0, 0), thickness=2)
            # draw eyelids
            for i in range(1, 16):
                center = org_eye[0, i, 0:2].astype(np.int)
                aligned_facial_img = cv2.circle(aligned_facial_img, (center[0], center[1]), radius=0, color=(255, 255, 100), thickness=2)
            # draw contour
            h_diameter = np.linalg.norm(org_iris[1,:] - org_iris[3,:])
            v_diameter = np.linalg.norm(org_iris[2,:] - org_iris[4,:])
            # h_diameters.append(h_diameter)
            # v_diameters.append(v_diameter)
            pupil_center = org_iris[0,0:2].astype(np.int)
            aligned_facial_img = cv2.circle(aligned_facial_img, (pupil_center[0], pupil_center[1]), radius=round(min(h_diameter, v_diameter)/2), color=(0, 0, 255), thickness=1)
        # save image
        img = cv2.cvtColor(aligned_facial_img, cv2.COLOR_RGB2BGR)
        return img