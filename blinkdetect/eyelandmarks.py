import argparse
import os
import shutil
import cv2
import torch
import numpy as np
import pandas as pd
from retinaface import RetinaFace
from retinaface.commons.postprocess import alignment_procedure
import matplotlib.pyplot as plt
import tqdm
from typing import List, Union

gpu0 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
gpu1 = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

from blinkdetect.models.facemesh import FaceMesh
from blinkdetect.models.irislandmarks import IrisLandmarks
from blinkdetect.metrics.distance import eyelids_directed_hausdorff, iris_diameter
from blinkdetect.tracking.optical_flow import compute_optFlow
from blinkdetect.image.analyze import color_analysis
from blinkdetect.image.misc import cut_region

FACE_MESH_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "facemesh.pth")
IRIS_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "irislandmarks.pth")

lower_right = [33, 7, 163, 144, 145, 153, 154, 155]
upper_right = [133, 173, 157, 158, 159, 160, 161, 246]
lower_left = [362, 382, 381, 380, 374, 373, 390, 249]
upper_left = [263, 466, 388, 387, 386, 385, 384, 398]
eye_corners_right = [lower_right[0], upper_right[0]]
eye_corners_left = [lower_left[0], upper_left[0]]
eye_corners = {"left":eye_corners_left, "right":eye_corners_right}

eye_corners = [lower_right[0], upper_right[0]]
left_eye_indices = lower_left + upper_left
right_eye_indices = lower_right + upper_right
eye_indices = {"left": left_eye_indices, "right": right_eye_indices}


def points_in_between(eye: np.ndarray) -> np.ndarray:
    """compute the positions of the middle curve between the two eyelids

    Args:
        eye (np.ndarray): [description]

    Returns:
        np.ndarray: shape of (N, 2)
    """
    frm = 1
    to = 16
    x, y = eye[:, frm:to, 0].reshape(-1), eye[:, frm:to, 1].reshape(-1)
    eye_corners = [tuple(eye[:, 0,0:2].reshape(-1)), tuple(eye[:, 8,0:2].reshape(-1))]

    midx = [eye_corners[0][0]]
    midy = [eye_corners[0][1]]
    midx += [np.mean([x[i], x[i+8]]) for i in range(7)]
    midy += [np.mean([y[i], y[i+8]]) for i in range(7)]
    midx += [eye_corners[1][0]]
    midy += [eye_corners[1][1]]

    return np.array([midx, midy]).T


class IrisHandler(object):
    """
    uses media pipe to draw iris center, iris contour and eyelids landmarks
    """
    def __init__(
        self,
        facemesh_path: str = FACE_MESH_MODEL_PATH, iris_path: str = IRIS_MODEL_PATH,
        face_detector: bool=True):
        super(IrisHandler, self).__init__()

        # requires BGR
        if face_detector:
            self.model = RetinaFace.build_model()
        else:
            self.model == None

        self._expansion_ratio = 0.25

        # requires RGB
        self.facemesh_net = FaceMesh().to(gpu0)
        self.facemesh_net.load_weights(facemesh_path)

        # requires RGB
        self.iris_net = IrisLandmarks().to(gpu1)
        self.iris_net.load_weights(iris_path)


    def expand_region(self, bbox_org: List, img: np.ndarray):
        """Expand bounding box by 25% of its size

        Args:
            bbox_org (List): [description]
            img (np.ndarray): [description]

        Returns:
            [type]: [description]
        """
        H = bbox_org[3] - bbox_org[1]
        W = bbox_org[2] - bbox_org[0]

        up = int(bbox_org[1] - self._expansion_ratio * H)
        down = int(bbox_org[3] + self._expansion_ratio * H)
        left = int(bbox_org[0] - self._expansion_ratio * W)
        right = int(bbox_org[2] + self._expansion_ratio * W)

        up = up if up > 0 else 0
        down = down if down < img.shape[0] else img.shape[0]
        left = left if left > 0 else 0
        right = right if right < img.shape[1] else img.shape[1]
        bbox_m = [left, up, right, down]

        return bbox_m


    def extract_eye_landmarks(self, face: np.ndarray, flip_right: bool=False):
        """previously extract_eye_region_curve. Extract
        eyelids landmarks and iris.

        Args:
            face (np.ndarray): horizontally aligned face

        Returns:
            dict: eye and iris landmarks for both eyes
        """
        resp = dict.fromkeys(eye_indices.keys())

        height, width = 192, 192
        img = cv2.resize(face, (width, height))
        detections = self.facemesh_net.predict_on_image(img).cpu().numpy()
        # get eye markers
        _detections = {"left": None, "right": None}
        detections_left = np.array(
            list(map(detections.__getitem__, left_eye_indices))
            )
        detections_right = np.array(
            list(map(detections.__getitem__, right_eye_indices))
            )
        _detections["left"] = detections_left
        _detections["right"] = detections_right

        for _eye in eye_indices:

            # eye center
            # compute the center and enlarge
            eye_center=np.mean(_detections[_eye],axis=0)
            x1=eye_center[0]
            y1=eye_center[1]

            # enlarge
            # check if the center is at the edges of the image
            _enlarge = 40
            _enlarge = min([x1, width-x1, y1, height-y1,_enlarge])
            left = int(x1 - _enlarge)
            right = int(x1 + _enlarge)
            top = int(y1 - _enlarge)
            bottom = int(y1 + _enlarge)
            eye_region_org = img[top:bottom,left:right, :]

            # iris landmarks detection
            # get the eye region
            eye_region = eye_region_org.copy()
            eye_region = cv2.resize(eye_region, (64, 64))

            if flip_right and _eye == "right":
                # flip
                eye_region = cv2.flip(eye_region, 1)

            eye_gpu, iris_gpu = self.iris_net.predict_on_image(eye_region)
            eye = eye_gpu.cpu().numpy()
            iris = iris_gpu.cpu().numpy()

            # scaling back
            e_H, e_W, _ = eye_region_org.shape
            e_scale_y, e_scale_x = e_H/64, e_W/64
            f_H, f_W, _ = face.shape
            f_scale_y, f_scale_x = f_H/192, f_W/192

            # original sizes
            org_iris = ((iris[0,:, :2] @ np.array([[e_scale_x, 0],[0,e_scale_y] ])) + np.array([left, top])) @ np.array([[f_scale_x, 0],[0,f_scale_y]])
            org_eye = ((eye[:,:,:2] @ np.array([[e_scale_x, 0],[0,e_scale_y] ])) + np.array([left, top])) @ np.array([[f_scale_x, 0],[0,f_scale_y]])
            eye = org_eye.copy()
            iris = org_iris.copy()
            
            resp[_eye] = [eye, iris]

        return resp


    def predict_eye_region(
        self,
        images_paths: list,
        facesInfo: pd.DataFrame,
        frames_exception: list=[]):
        """
        [depricated summary]
            uses mediapipe face mesh, eyelids and iris estimators to get the color values alongside
            the lines segments connecting the center of the iris with the two corners of the eye as a timeseries from the left corner of the eye
        """

        outputDF = facesInfo.copy()
        if 'eyelids_dist_right' not in outputDF:
            outputDF['eyelids_dist_right'] = None
            outputDF['eyelids_dist_left'] = None
            outputDF['mean_color_right'] = None
            outputDF['mean_color_left'] = None
            outputDF['std_left'] = None
            outputDF['std_right'] = None
            outputDF['line_points_left'] = None
            outputDF['line_points_right'] = None


        processed_images = []
        frames_names = []

        for _img_path in tqdm.tqdm(images_paths, total=len(images_paths), leave=False, desc="frame"):

            img_name = os.path.basename(_img_path)
            _name, _ = img_name.split(".")

            # REMOVE THIS AND SOLVE IT OUTSIDE
            # check if it is not intended to be processed
            if _name in frames_exception:
                continue

            # REMOVE THIS AND SOLVE IT OUTSIDE
            if _name not in outputDF.index:
                continue

            # REMOVE THIS AND SOLVE IT OUTSIDE
            # add path and frames number
            processed_images.append(_img_path)
            frames_names.append(_name)

            # 1. EXTRACT FACE
            # bounding box
            bbox = list(outputDF.loc[_name][['left', 'top', 'right', 'bottom']].astype(np.int32))
            left_eye = list(outputDF.loc[_name][['left_eye_x', 'left_eye_y']].astype(np.float64))
            right_eye = list(outputDF.loc[_name][['right_eye_x', 'right_eye_y']].astype(np.float64))
            nose = list(outputDF.loc[_name][['nose_x', 'nose_y']].astype(np.float64))

            # read image
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # margin
            bbox_m = self.expand_region(bbox_org, img)
            # crop face
            status, facial_img = cut_region(_img_path, bbox)

            if not status:
                outputDF.loc[_name]['faces_not_found'] = 1
                outputDF.loc[_name]['faces_number'] = 0
                continue

            # 2. ALIGN IT 
            aligned_facial_img = postprocess.alignment_procedure(facial_img, right_eye, left_eye)#, nose)

            # 3. EXTRACT
            # face mesh prediction
            # img_facemesh = aligned_facial_img.copy()
            resp = self.extract_eye_landmarks(face=aligned_facial_img)


            # REMOVE THIS AND SOLVE IT OUTSIDE
            # # # # # # # # # # # # # # #
            # Analysis of color change  #
            # # # # # # # # # # # # # # # 
            for _eye in resp:
                eye = resp[_eye]
                iris = resp[_eye]

                mid_points = points_in_between(eye)

                _std, _mean = color_analysis(aligned_facial_img, mid_points)
                _eyelids_dist = eyelids_directed_hausdorff(set1_indices=[1,8], set2_indices=[9,16], landmarks=eye)
                resp[_eye]['line_points'] = []
                resp[_eye]['std'] = _std
                resp[_eye]['mean_color'] = _mean
                resp[_eye]['eyelids_dist'] = _eyelids_dist

                outputDF.loc[_name][f'eyelids_dist_{_eye}'] = resp[_eye]['eyelids_dist']
                outputDF.loc[_name][f'mean_color_{_eye}'] = resp[_eye]['mean_color']
                outputDF.loc[_name][f'std_{_eye}'] = resp[_eye]['std']
                outputDF.loc[_name][f'line_points_{_eye}'] = resp[_eye]['line_points']

        return outputDF, processed_images, frames_names#, _means, _stds, _eyelids_dists


    def analyze_eye_region(self, image_path: str, facial_area, landmarks):
        """
            uses mediapipe face mesh, eyelids and iris estimators to get the following:
                - eyelids distance for both eyes
                - std and mean for the curve between the upper and lower eyelids
        """
        # read image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 1. EXTRACT FACE
        bbox_org = facial_area
        left_eye = landmarks['left_eye']
        right_eye = landmarks['right_eye']
        # crop margined face
        bbox_m = self.expand_region(bbox_org, img)
        # currently it is assumed that there is a face --> success==True
        success, facial_img = cut_region(_img_path, bbox)

        # 2. ALIGN IT 
        aligned_facial_img = postprocess.alignment_procedure(facial_img, right_eye, left_eye)

        # 3. EXTRACT
        # face mesh prediction
        resp = self.extract_eye_landmarks(face=aligned_facial_img)

        # # # # # # # # # # # # # # #
        # Analysis of color change  #
        # # # # # # # # # # # # # # # 
        final_reponse = dict()
        for _eye in resp:
            eye = resp[_eye]
            iris = resp[_eye]

            mid_points = points_in_between(eye)
            std, mean = color_analysis(aligned_facial_img, mid_points)
            eyelids_dist = eyelids_directed_hausdorff(set1_indices=[1,8], set2_indices=[9,16], landmarks=eye)

            final_reponse[f'eyelids_dist_{_eye}'] = std
            final_reponse[f'mean_color_{_eye}'] = mean
            final_reponse[f'std_color_{_eye}'] = eyelids_dist
    
        return final_reponse


    def overlay_image(self, img: np.ndarray, transform = False, eyelids = True, iris = True):
        """This expects input to be RGB if transform is False.
        if transform is True, It will convert the input into RGB

        Args:
            image (np.ndarray): [description]
            transform (bool, optional): [description]. Defaults to False.
        """
        assert self.model is not None, "this version does not support other bounding boxes formats"

        if not transform:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        faces = RetinaFace.detect_faces(img, model=self.model)
        if type(faces) is tuple:
            return img
        # in case there is a face take the first one
        bbox_org = faces['face_1']['facial_area']
        left_eye_pt = faces['face_1']['landmarks']['left_eye']
        right_eye_pt = faces['face_1']['landmarks']['right_eye']

        # margin
        bbox_m = self.expand_region(bbox_org, img)

        # cut face
        sucess, facial_img_m = cut_region(img, bbox_m)

        # alignment
        aligned_facial_img = alignment_procedure(facial_img_m, left_eye_pt, right_eye_pt)

        if transform:
            aligned_facial_img = cv2.cvtColor(aligned_facial_img, cv2.COLOR_BGR2RGB)
        # eyes landmarks
        resp = self.extract_eye_landmarks(aligned_facial_img)
        # 
        aligned_facial_img_eyelids = aligned_facial_img.copy()
        aligned_facial_img_iris = aligned_facial_img.copy()

        for eye_key in resp:
            org_eye, org_iris = resp[eye_key]

            # draw eyelids
            if eyelids:
                
                for i in range(1, 16):
                    center = org_eye[0, i, 0:2].astype(np.int)
                    aligned_facial_img_eyelids = cv2.circle(aligned_facial_img_eyelids, (center[0], center[1]), radius=0, color=(255, 255, 100), thickness=2)

            # draw iris landmarks
            if iris:
                for i in range(5):
                    center = org_iris[i,0:2].astype(np.int)
                    aligned_facial_img_iris = cv2.circle(aligned_facial_img_iris, (center[0], center[1]), radius=0, color=(255, 0, 0), thickness=2)

                # draw pupil circle
                h_diameter = np.linalg.norm(org_iris[1,:] - org_iris[3,:])
                v_diameter = np.linalg.norm(org_iris[2,:] - org_iris[4,:])
                # h_diameters.append(h_diameter)
                # v_diameters.append(v_diameter)
                pupil_center = org_iris[0,0:2].astype(np.int)
                aligned_facial_img_iris = cv2.circle(aligned_facial_img_iris, (pupil_center[0], pupil_center[1]), radius=round(min(h_diameter, v_diameter)/2), color=(0, 0, 255), thickness=1)

        aligned_facial_img = np.hstack([aligned_facial_img, aligned_facial_img_iris, aligned_facial_img_eyelids])
        return aligned_facial_img


    def analyze(self, img_input: np.ndarray, prev_img: np.ndarray = None, transform = False, eyelids = True, iris = True):
        """
        This expects input to be RGB if transform is False.
        if transform is True, It will convert the input into RGB

        Args:
            img (np.ndarray): [description]
            transform (bool, optional): Defaults to False.
            eyelids (bool, optional): whether to draw eyelids or not. Defaults to True.
            iris (bool, optional): whether to draw iris or not. Defaults to True.

        Returns:
            [type]: [description]
        """

        assert self.model is not None, "this version does not support other bounding boxes formats"

        if not transform:
            img = cv2.cvtColor(img_input, cv2.COLOR_RGB2BGR)
        else:
            img = img_input.copy()

        # compute optical flow
        mask_OF = compute_optFlow(prev_img, img)

        faces = RetinaFace.detect_faces(img, model=self.model)
        if type(faces) is tuple:
            return img
        # in case there is a face take the first one
        bbox_org = faces['face_1']['facial_area']
        left_eye_pt = faces['face_1']['landmarks']['left_eye']
        right_eye_pt = faces['face_1']['landmarks']['right_eye']

        # margin
        bbox_m = self.expand_region(bbox_org, img)

        # cut face
        sucess, facial_img_m = cut_region(img, bbox_m)
        _, mask_OF = cut_region(mask_OF, bbox_m)

        # alignment
        aligned_facial_img = alignment_procedure(facial_img_m, right_eye_pt, left_eye_pt)

        if transform:
            aligned_facial_img = cv2.cvtColor(aligned_facial_img, cv2.COLOR_BGR2RGB)
        # eyes landmarks
        resp = self.extract_eye_landmarks(aligned_facial_img)
        # 
        aligned_facial_img_eyelids = aligned_facial_img.copy()
        aligned_facial_img_iris = aligned_facial_img.copy()

        eyelidsDistances = dict.fromkeys(resp.keys())
        irisesDiameters = dict.fromkeys(resp.keys())
        pupil2corners = dict.fromkeys(resp.keys())

        for eye_key in resp:
            org_eye, org_iris = resp[eye_key]

            eyelidsDistances[eye_key] = eyelids_directed_hausdorff(set1_indices=[1,8], set2_indices=[9,16], landmarks=org_eye)
            irisesDiameters[eye_key] = iris_diameter(org_iris)


            # draw eyelids
            if eyelids:
                
                for i in range(1, 16):
                    center = np.rint(org_eye[0, i, 0:2]).astype(np.int)
                    aligned_facial_img_eyelids = cv2.circle(aligned_facial_img_eyelids, (center[0], center[1]), radius=0, color=(255, 255, 100), thickness=1)

            # distance from puipl center to the left corner
            pupil2corners[eye_key] = np.linalg.norm(org_iris[0,0:2] - org_eye[0, 0, 0:2])
            # pupil_center = np.rint(org_iris[0,0:2]).astype(np.int)
            corner_left = np.rint(org_eye[0, 0, 0:2]).astype(np.int)
            corner_right = np.rint(org_eye[0, 8, 0:2]).astype(np.int)
            aligned_facial_img_eyelids = cv2.circle(aligned_facial_img_eyelids, (corner_left[0], corner_left[1]), radius=0, color=(255, 0, 0), thickness=2)
            aligned_facial_img_eyelids = cv2.circle(aligned_facial_img_eyelids, (corner_right[0], corner_right[1]), radius=0, color=(255, 0, 0), thickness=2)
            # eye_corners = [tuple(org_eye[:, 0,0:2].reshape(-1)), tuple(org_eye[:, 8,0:2].reshape(-1))]
            # draw iris landmarks
            if iris:
                # for i in range(5):
                #     center = org_iris[i,0:2].astype(np.int)
                #     aligned_facial_img_iris = cv2.circle(aligned_facial_img_iris, (center[0], center[1]), radius=0, color=(255, 0, 0), thickness=2)

                # draw pupil circle
                h_diameter = np.linalg.norm(org_iris[1,:] - org_iris[3,:])
                v_diameter = np.linalg.norm(org_iris[2,:] - org_iris[4,:])
                # h_diameters.append(h_diameter)
                # v_diameters.append(v_diameter)
                pupil_center = np.rint(org_iris[0,0:2]).astype(np.int)
                aligned_facial_img_iris = cv2.circle(aligned_facial_img_iris, (pupil_center[0], pupil_center[1]), radius=round((h_diameter + v_diameter) / 4), color=(0, 0, 255), thickness=0)

        img = cv2.cvtColor(facial_img_m, cv2.COLOR_BGR2RGB)
        aligned_facial_img = np.hstack([img, aligned_facial_img_iris, aligned_facial_img_eyelids, mask_OF])

        return aligned_facial_img, eyelidsDistances, irisesDiameters, pupil2corners


    def extract_features(self, img_input: np.ndarray, transform = False):
        """
        This expects input to be RGB if transform is False.
        if transform is True, It will convert the input into RGB

        Args:
            img (np.ndarray): [description]
            transform (bool, optional): Defaults to False.

        Returns:
            [type]: [description]
        """

        assert self.model is not None, "this version does not support other bounding boxes formats"

        if not transform:
            img = cv2.cvtColor(img_input, cv2.COLOR_RGB2BGR)
        else:
            img = img_input.copy()

        faces = RetinaFace.detect_faces(img, model=self.model)
        if type(faces) is tuple:
            return img
        # in case there is a face take the first one
        bbox_org = faces['face_1']['facial_area']
        left_eye_pt = faces['face_1']['landmarks']['left_eye']
        right_eye_pt = faces['face_1']['landmarks']['right_eye']

        # margin
        bbox_m = self.expand_region(bbox_org, img)

        # cut face
        sucess, facial_img_m = cut_region(img, bbox_m)

        # alignment
        aligned_facial_img = alignment_procedure(facial_img_m, right_eye_pt, left_eye_pt)

        if transform:
            aligned_facial_img = cv2.cvtColor(aligned_facial_img, cv2.COLOR_BGR2RGB)
        # eyes landmarks
        resp = self.extract_eye_landmarks(aligned_facial_img, flip_right=True)

        eyelidsDistances = dict.fromkeys(resp.keys())
        irisesDiameters = dict.fromkeys(resp.keys())
        pupil2corners = dict.fromkeys(resp.keys())

        for eye_key in resp:
            org_eye, org_iris = resp[eye_key]

            # eyelids distance
            eyelidsDistances[eye_key] = eyelids_directed_hausdorff(set1_indices=[1,8], set2_indices=[9,16], landmarks=org_eye)

            # iris diameter
            irisesDiameters[eye_key] = iris_diameter(org_iris)

            # distance from puipl center to the left corner (inner corner)
            pupil2corners[eye_key] = np.linalg.norm(org_iris[0,0:2] - org_eye[0, 0, 0:2])


        return eyelidsDistances, irisesDiameters, pupil2corners