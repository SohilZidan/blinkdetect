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
# print("PyTorch version:", torch.__version__)
# print("CUDA version:", torch.version.cuda)
# print("cuDNN version:", torch.backends.cudnn.version())
# print("OpenCV version:", cv2.__version__)
# torch.cuda.device_count()
gpu0 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
gpu1 = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

from blinkdetect.models.facemesh import FaceMesh
from blinkdetect.models.irislandmarks import IrisLandmarks
from blinkdetect.metrics.distance import eyelids_directed_hausdorff, iris_diameter

FACE_MESH_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "facemesh.pth")
IRIS_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "irislandmarks.pth")

lower_right = [33, 7, 163, 144, 145, 153, 154, 155]
upper_right = [133, 173, 157, 158, 159, 160, 161, 246]
lower_left = [362, 382, 381, 380, 374, 373, 390, 249]
upper_left = [263, 466, 388, 387, 386, 385, 384, 398]

eye_corners = [lower_right[0], upper_right[0]]
left_eye_indices = lower_left + upper_left
right_eye_indices = lower_right + upper_right
eye_indices = {"left": left_eye_indices, "right": right_eye_indices}

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


    def expand_bbox(self, bbox_org: List, img: np.ndarray):
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


    def cut_face(self, img_path: Union[np.ndarray, str], bbox: List) -> np.ndarray:
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

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # cut the face
        facial_img = img[bbox[1]:bbox[3], bbox[0]: bbox[2]]
        success = (facial_img.size > 0)

        return success, facial_img


    def extract_eye_landmarks(self, face: np.ndarray):
        """previously extract_eye_region_curve

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

            # iris landmarks
            # iris detection
            # get the eye region
            eye_region = eye_region_org.copy()
            eye_region = cv2.resize(eye_region, (64, 64))

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

            # # # # # #
            # eyelids #
            # # # # # #
        #     frm = 1
        #     to = 16
        #     x, y = eye[:, frm:to, 0].reshape(-1), eye[:, frm:to, 1].reshape(-1)
        #     # eye_corners = [(int(x[0]),int(y[0])), (int(x[-1]),int(y[-1]))]
        #     eye_corners = [tuple(eye[:, 0,0:2].reshape(-1)), tuple(eye[:, 8,0:2].reshape(-1))]

        #     midx = [eye_corners[0][0]]
        #     midy = [eye_corners[0][1]]
        #     midx += [np.mean([x[i], x[i+8]]) for i in range(7)]
        #     midy += [np.mean([y[i], y[i+8]]) for i in range(7)]
        #     midx += [eye_corners[1][0]]
        #     midy += [eye_corners[1][1]]

        #     # # # # #
        #     # iris  #
        #     # # # # #
        #     x_pupil_center, y_pupil_center ,_ = iris[0, 0]
        #     pupil_center=(int(x_pupil_center), int(y_pupil_center))

        #     resp[_eye] = {"eye_region": eye_region, "eye_corners":eye_corners, "pupil_center":pupil_center, "eye": eye[:, 0:16, :], "midx": midx, 'midy':midy, "iris": iris}

        return resp

    
    def points_in_between(self, eye: np.ndarray) -> np.ndarray:
        """compute the middle curve between the two eyelids

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


    def color_analysis_curve(self, eye_region: np.ndarray, mid_points: np.ndarray):
        # mid_points = np.stack([midx,midy], axis=1)
        diff_mid_points = mid_points[1:] - mid_points[:-1]
        _nums = np.ceil(np.linalg.norm(diff_mid_points, axis=1)).astype(np.int32)

        acc_x = np.array([])
        acc_y = np.array([])
        for i in range(len(diff_mid_points)):
            num_t = _nums[i]
            acc_x = np.hstack((acc_x, np.linspace(midx[i], midx[i+1], num_t)))
            acc_y = np.hstack((acc_y, np.linspace(midy[i], midy[i+1], num_t)))

        x, y = acc_x, acc_y

        # eyelids distance
        _eyelids_dist = eyelids_directed_hausdorff(set1_indices=[1,8], set2_indices=[9,16], landmarks=eye, iris=iris)

        _std = []
        _mean = []
        for _dim in range(eye_region.shape[2]):
            z = eye_region[:,:,_dim].copy()
            zi = scipy.ndimage.map_coordinates(z, np.vstack((y,x)))
            # measurements  
            _std.append(np.std(zi))
            _mean.append(np.mean(zi))

        return _std, _mean, _eyelids_dist


    def predict_eye_region(
        self,
        images_paths: list,
        facesInfo: pd.DataFrame,
        frames_exception: list=[]):#,
        # save_faces:str="",
        # save_eye_change:str=""):
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
            # 
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
            bbox_m = self.expand_bbox(bbox_org, img)
            # crop face
            status, facial_img = self.cut_face(_img_path, bbox)

            if not status:
                outputDF.loc[_name]['faces_not_found'] = 1
                outputDF.loc[_name]['faces_number'] = 0
                continue

            # 2. ALIGN IT 
            aligned_facial_img = postprocess.alignment_procedure(facial_img, right_eye, left_eye)#, nose)

            # 3. EXTRACT
            # face mesh prediction
            img_facemesh = aligned_facial_img.copy()
            resp = self.extract_eye_landmarks(face=img_facemesh)


            # REMOVE THIS AND SOLVE IT OUTSIDE
            # # # # # # # # # # # # # # #
            # Analysis of color change  #
            # # # # # # # # # # # # # # # 
            for _eye in resp:
                # eye = resp[_eye]['eye']
                iris = resp[_eye]['eye']
                min_coord = np.min(eye.reshape(-1, 3)[:, :2], axis=0)
                max_coord = np.max(eye.reshape(-1, 3)[:, :2], axis=0)
                X, Y = min_coord.astype(np.int)
                X1, Y1 = max_coord.astype(np.int)
                eye_region = aligned_facial_img[Y:Y1, X:X1, :]

                #
                mid_points = self.points_in_between(eye)
                #
                _std, _mean, _eyelids_dist = self.color_analysis_curve(eye_region, mid_points)
                resp[_eye]['line_points'] = []
                resp[_eye]['std'] = _std
                resp[_eye]['mean_color'] = _mean
                resp[_eye]['eyelids_dist'] = _eyelids_dist
                
                outputDF.loc[_name][f'eyelids_dist_{_eye}'] = resp[_eye]['eyelids_dist']
                outputDF.loc[_name][f'mean_color_{_eye}'] = resp[_eye]['mean_color']
                outputDF.loc[_name][f'std_{_eye}'] = resp[_eye]['std']
                outputDF.loc[_name][f'line_points_{_eye}'] = resp[_eye]['line_points']

        return outputDF, processed_images, frames_names#, _means, _stds, _eyelids_dists


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
        bbox_m = self.expand_bbox(bbox_org, img)

        # cut face
        sucess, facial_img_m = self.cut_face(img, bbox_m)

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


    def compute_optFlow(self, prev_gray: np.ndarray, current_frame: np.ndarray):
        mask = np.zeros_like(current_frame)
        # Sets image saturation to maximum
        mask[..., 1] = 255
        if prev_gray is None:
            return mask

        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        # Calculates dense optical flow by Farneback method
        flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, 
                                        None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)

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


    def analyze(self, img: np.ndarray, prev_img: np.ndarray = None, transform = False, eyelids = True, iris = True):
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
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # compute optical flow
        if prev_img is None:
            prev_gray = prev_img
        else:
            prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask_OF = self.compute_optFlow(prev_gray, img)

        faces = RetinaFace.detect_faces(img, model=self.model)
        if type(faces) is tuple:
            return img
        # in case there is a face take the first one
        bbox_org = faces['face_1']['facial_area']
        left_eye_pt = faces['face_1']['landmarks']['left_eye']
        right_eye_pt = faces['face_1']['landmarks']['right_eye']

        # margin
        bbox_m = self.expand_bbox(bbox_org, img)

        # cut face
        sucess, facial_img_m = self.cut_face(img, bbox_m)
        _, mask_OF = self.cut_face(mask_OF, bbox_m)

        # alignment
        aligned_facial_img = alignment_procedure(facial_img_m, left_eye_pt, right_eye_pt)

        if transform:
            aligned_facial_img = cv2.cvtColor(aligned_facial_img, cv2.COLOR_BGR2RGB)
        # eyes landmarks
        resp = self.extract_eye_landmarks(aligned_facial_img)
        # 
        aligned_facial_img_eyelids = aligned_facial_img.copy()
        aligned_facial_img_iris = aligned_facial_img.copy()

        eyelidsDistances = dict.fromkeys(resp.keys())
        irisesDiameters = dict.fromkeys(resp.keys())

        for eye_key in resp:
            org_eye, org_iris = resp[eye_key]

            eyelidsDistances[eye_key] = eyelids_directed_hausdorff(set1_indices=[1,8], set2_indices=[9,16], landmarks=org_eye)
            irisesDiameters[eye_key] = iris_diameter(org_iris)

            # draw eyelids
            if eyelids:
                
                for i in range(1, 16):
                    center = org_eye[0, i, 0:2].astype(np.int)
                    aligned_facial_img_eyelids = cv2.circle(aligned_facial_img_eyelids, (center[0], center[1]), radius=0, color=(255, 255, 100), thickness=1)

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
                pupil_center = org_iris[0,0:2].astype(np.int)
                aligned_facial_img_iris = cv2.circle(aligned_facial_img_iris, (pupil_center[0], pupil_center[1]), radius=round(min(h_diameter, v_diameter)/2), color=(0, 0, 255), thickness=0)

        aligned_facial_img = np.hstack([aligned_facial_img, aligned_facial_img_iris, aligned_facial_img_eyelids, mask_OF])

        return aligned_facial_img, eyelidsDistances, irisesDiameters
