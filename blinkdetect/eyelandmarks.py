import os
import cv2
import torch
import numpy as np
from retinaface import RetinaFace
from retinaface.commons.postprocess import alignment_procedure
from typing import List, Union

gpu0 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
gpu1 = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

from blinkdetect.models.facemesh import FaceMesh
from blinkdetect.models.irislandmarks import IrisLandmarks
from blinkdetect.metrics.distance import eyelids_directed_hausdorff, iris_diameter
from blinkdetect.tracking.optical_flow import compute_optFlow
from blinkdetect.image.misc import cut_region, expand_region

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


class IrisHandler(object):
    """
    uses media pipe to extract eye landmarks
    in a horizontally aligned face image
    """
    def __init__(
        self,
        facemesh_path: str = FACE_MESH_MODEL_PATH,
        iris_path: str = IRIS_MODEL_PATH):
        # face_detector: bool=True):
        super(IrisHandler, self).__init__()

        # requires BGR
        # if face_detector:
        #     self.model = RetinaFace.build_model()
        # else:
        #     self.model = None

        self._expansion_ratio = 0.25

        # requires RGB
        self.facemesh_net = FaceMesh().to(gpu0)
        self.facemesh_net.load_weights(facemesh_path)

        # requires RGB
        self.iris_net = IrisLandmarks().to(gpu1)
        self.iris_net.load_weights(iris_path)


    def extract_eye_landmarks(self, face: np.ndarray, flip_right: bool=False, transform: bool=False):
        """
        Extract eyelids landmarks and iris.

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

            if flip_right and (_eye == "right"):
                # flip
                eye_region = cv2.flip(eye_region, 1)

            eye_gpu, iris_gpu = self.iris_net.predict_on_image(eye_region)
            eye = eye_gpu.cpu().numpy()
            iris = iris_gpu.cpu().numpy()

            if transform:
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
            else:
                eye = eye[:,:,:2]
                iris = eye[0,:,:2]
            
            resp[_eye] = [eye, iris]

        return resp


    def overlay_image(
        self, img: np.ndarray, transform = False,
        eyelids = True, iris = True,
        face_landmarks = None):
        """This expects input to be RGB if transform is False.
        if transform is True, It will convert the input into RGB

        Args:
            image (np.ndarray): [description]
            transform (bool, optional): [description]. Defaults to False.
        """
        # assert self.model is not None, "this version does not support other bounding boxes formats"

        if not transform:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # if face_landmarks is None:
        #     print("image shape:",img.shape)
        #     faces = RetinaFace.detect_faces(img, model=self.model)
        #     if type(faces) is tuple:
        #         return None
        #     # in case there is a face take the first one
        #     bbox_org = faces['face_1']['facial_area']
        #     left_eye_pt = faces['face_1']['landmarks']['left_eye']
        #     right_eye_pt = faces['face_1']['landmarks']['right_eye']
        # else:
        bbox_org = face_landmarks['facial_area']
        left_eye_pt = face_landmarks['landmarks']['left_eye']
        right_eye_pt = face_landmarks['landmarks']['right_eye']

        # margin
        bbox_m = expand_region(bbox_org, img)

        # cut face
        sucess, facial_img_m = cut_region(img, bbox_m)

        # alignment
        aligned_facial_img = alignment_procedure(facial_img_m, right_eye_pt, left_eye_pt)

        if transform:
            aligned_facial_img = cv2.cvtColor(aligned_facial_img, cv2.COLOR_BGR2RGB)
        # eyes landmarks
        resp = self.extract_eye_landmarks(aligned_facial_img)
        aligned_facial_img = cv2.cvtColor(aligned_facial_img, cv2.COLOR_RGB2BGR)
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


    def analyze(
        self, img_input: np.ndarray, prev_img: np.ndarray = None,
        transform = False, eyelids = True, iris = True, face_landmarks = None):
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

        if face_landmarks is None:
            faces = RetinaFace.detect_faces(img, model=self.model)
            if type(faces) is tuple:
                return img
            # in case there is a face take the first one
            bbox_org = faces['face_1']['facial_area']
            left_eye_pt = faces['face_1']['landmarks']['left_eye']
            right_eye_pt = faces['face_1']['landmarks']['right_eye']
        else:
            bbox_org = face_landmarks['facial_area']
            left_eye_pt = face_landmarks['landmarks']['left_eye']
            right_eye_pt = face_landmarks['landmarks']['right_eye']

        # margin
        bbox_m = expand_region(bbox_org, img)

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


    def extract_features(self, img_input: np.ndarray, transform = False, face_landmarks = None):
        """
        This expects input to be RGB if transform is False.
        if transform is True, It will convert the input into RGB

        Args:
            img (np.ndarray): [description]
            transform (bool, optional): Defaults to False.

        Returns:
            [type]: [description]
        """

        # assert self.model is not None, "this version does not support other bounding boxes formats"

        if not transform:
            img = cv2.cvtColor(img_input, cv2.COLOR_RGB2BGR)
        else:
            img = img_input.copy()


        if face_landmarks is None:
            faces = RetinaFace.detect_faces(img, model=self.model)
            if type(faces) is tuple:
                return None
            # in case there is a face take the first one
            bbox_org = faces['face_1']['facial_area']
            left_eye_pt = faces['face_1']['landmarks']['left_eye']
            right_eye_pt = faces['face_1']['landmarks']['right_eye']
        else:
            bbox_org = face_landmarks['facial_area']
            left_eye_pt = face_landmarks['landmarks']['left_eye']
            right_eye_pt = face_landmarks['landmarks']['right_eye']

        # margin
        bbox_m = expand_region(bbox_org, img)

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