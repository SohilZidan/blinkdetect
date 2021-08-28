#!/usr/bin/env python3
# coding: utf-8

from genericpath import exists
import os
import glob
import shutil
from math import ceil
import argparse
import numpy as np
import pandas as pd
import cv2
from pandas.core.reshape.concat import concat
import torch
import scipy
import tqdm
from retinaface import RetinaFace
from facemesh import FaceMesh
from irislandmarks import IrisLandmarks

import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff
from retinaface.commons import postprocess

print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())
print("OpenCV version:", cv2.__version__)

gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 
FACE_MESH_MODEL_PATH=os.path.join(os.path.dirname(__file__), "..","models","facemesh.pth")
IRIS_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..","models","irislandmarks.pth")
# 
dataset_root = os.path.join(os.path.dirname(__file__), "..", "dataset")

# 
# eye landmarks indices
lower_right=[362,382,381,380,374,373,390,249]
upper_right=[263,466,388,387,386,385,384,398]
lower_left=[23, 7, 163, 144, 145, 153, 154, 155]
upper_left=[133, 173, 157, 158, 159, 160, 161, 246]
# 
eye_corners = [lower_right[0], upper_right[0]]
left_eye_indices = lower_left+upper_left
right_eye_indices = lower_right + upper_right
eye_indices = {"left":left_eye_indices, "right":right_eye_indices}


def parser():
    _parser = argparse.ArgumentParser()
    _parser.add_argument('-pid', '--participant_id', required=True)
    _parser.add_argument('-rng', '--range', type=int, default=[0,-1], nargs=2)
    _parser.add_argument('--batch', type=int, default=32, help='number of frames to be saved as a batch')
    _parser.add_argument('--resume', action='store_true', help='if true existed frames of an existed participant will not be replaced')
    _parser.add_argument('--save_faces', type=str, default="", help='wheather to save faces or not')

    return _parser.parse_args()

def eyelids_directed_hausdorff(set1_indices: list, set2_indices: list, landmarks: np.ndarray):
    A = landmarks[:, set1_indices[0]:set1_indices[1], 0:2].reshape((-1,2))
    B = landmarks[:, set2_indices[0]:set2_indices[1], 0:2].reshape((-1,2))
    # print(" A:\n",A,'\n',"B:\n",B)

    return directed_hausdorff(B,A)[0]

def extract_eye_region(face: np.ndarray, facemeshnet, iris_net):
    """
    """
    resp = {}
    height, width = 192, 192
    img = cv2.resize(face, (width, height))
    detections = facemeshnet.predict_on_image(img).cpu().numpy()
    for _eye in eye_indices:

        # get eye markers
        _detections = np.array(list(map(detections.__getitem__, eye_indices[_eye])))
        # detections_left = np.array(list(map(detections.__getitem__, right_eye)))

        # compute the center and enlarge
        eye_center=np.mean(_detections,axis=0)
        x1=eye_center[0]
        y1=eye_center[1]
        # check if the center is at the edges of the image
        _enlarge = 64
        _enlarge = min([x1, width-x1, y1, height-y1,_enlarge])
        left=int(x1-_enlarge)
        right=int(x1+_enlarge)
        top=int(y1-_enlarge)
        bottom=int(y1+_enlarge)

        # iris detection
        # get the eye region
        eye_region = img[top:bottom,left:right, :]
        eye_region = cv2.resize(eye_region, (64, 64))

        eye_gpu, iris_gpu = iris_net.predict_on_image(eye_region)

        eye = eye_gpu.cpu().numpy()
        iris = iris_gpu.cpu().numpy()
        # # # # # #
        # eyelids #
        # # # # # #
        frm = 0
        to = 16
        x, y = eye[:, frm:to, 0].reshape(-1), eye[:, frm:to, 1].reshape(-1)
        eye_corners = [(int(x[0]),int(y[0])), (int(x[-1]),int(y[-1]))]
        # # # # #
        # iris  #
        # # # # #
        x_pupil_center, y_pupil_center ,_ = iris[0, 0]
        pupil_center=(int(x_pupil_center), int(y_pupil_center))

        resp[_eye] = {"eye_region": eye_region, "eye_corners":eye_corners, "pupil_center":pupil_center, "eye":eye[:, frm:to, :]}
        # eye_region, eye_corners, pupil_center, eye


    return resp

def color_analysis(eye_region, eye_corners, pupil_center, eye):
    tmp = np.array(eye_corners)
    # distance between the corners and the pupil center --> number of pixels
    num1 = np.ceil(np.linalg.norm(tmp[0,:] - np.array(pupil_center))).astype(np.int32)
    num2 = np.ceil(np.linalg.norm(np.array(pupil_center) - tmp[1,:])).astype(np.int32)
    # distance between the cornersof the eye --> number of pixels
    num = np.ceil(np.linalg.norm(tmp[0,:] - tmp[1,:])).astype(np.int32)

    #
    # z = cv2.cvtColor(eye_region, cv2.COLOR_RGB2GRAY)
    # 
    x0, y0 = eye_corners[0] # These are in _pixel_ coordinates!!
    x1, y1 = pupil_center
    x2, y2 = eye_corners[1]
    # eyelids distance
    _eyelids_dist = eyelids_directed_hausdorff(set1_indices=[0,8], set2_indices=[8,16], landmarks=eye)
    # 
    line_points = [x0, y0, x1, y1, x2, y2]

    # path passes through the pupil center
    x, y = np.hstack((np.linspace(x0, x1, num1), np.linspace(x1, x2, num2))), np.hstack((np.linspace(y0, y1, num1), np.linspace(y1, y2, num2)))
    _std = []
    _mean = []
    for _dim in range(eye_region.shape[2]):
        z = eye_region[:,:,_dim].copy()
        zi = scipy.ndimage.map_coordinates(z, np.vstack((y,x)))
        # measurements  
        _std.append(np.std(zi))
        _mean.append(np.mean(zi))
    return line_points, _std, _mean, _eyelids_dist

def saveEyeChange(eye_region, color_value, file_name, line_points, line_type: str="straight"):
    _std = np.std(color_value)
    x0, y0, x1, y1, x2, y2= line_points

    fig, axes = plt.subplots(nrows=3,ncols=2)
    _image = cv2.cvtColor(eye_region, cv2.COLOR_RGB2GRAY)
    plt.subplot(3, 2, 1)
    plt.imshow(_image, cmap='gray')
    if line_type == "straight":
        plt.plot([x0, x2], [y0, y2], 'ro-')
    else:
        plt.plot([x0, x1], [y0, y1], 'ro-')
        plt.plot([x1, x2], [y1, y2], 'bo-')
    plt.axis('image')

    plt.subplot(3, 2, 2)
    plt.imshow(eye_region, cmap='gray')
    plt.axis('image')

    plt.subplot(3, 1, 2)
    plt.title("color values")
    plt.plot(color_value,'.')#[:int(len(zi)/2)])
    plt.subplot(3, 1, 3)
    plt.title("color change")
    plt.plot(np.abs(color_value[1:].astype(np.int32)-color_value[:-1].astype(np.int32)), '*r')
    fig.tight_layout()
    plt.savefig(f"{file_name}_{line_type}_std{_std:.2f}.jpg", dpi=300, bbox_inches='tight')

def predict_eye_region(images_paths: list, facesInfo: pd.DataFrame, facemeshnet, iris_net, frames_exception: list=[], save_faces:str="", save_eye_change:str=""):
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

    # 
    _save_faces = False
    _save_eye_change = False
    if save_faces != "":
        _save_faces = True
        os.makedirs(save_faces, exist_ok=True)
    
    if save_eye_change != "":
        _save_eye_change = True
        os.makedirs(save_eye_change, exist_ok=True)
    
    
    # 
    # _eyelids_dists = {'right': [], "left": []}
    # _stds = {'right': [], "left": []}
    # _means = {'right': [], "left": []}
    # lines_points = {'right': [], "left": []}

    processed_images = []
    _frames_names = []


    _images = images_paths
    

    for _img_path in tqdm.tqdm(_images, total=len(_images)):
        # 
        img_name = os.path.basename(_img_path)
        _name, _ = img_name.split(".")

        # check if it is not intended to be processed
        if _name in frames_exception:
            continue

        if _name not in outputDF.index:
            continue

        # add path and frames number
        processed_images.append(_img_path)
        _frames_names.append(_name)

        # bounding box
        bbox = list(outputDF.loc[_name][['left', 'top', 'right', 'bottom']].astype(np.int32))
        left_eye = list(outputDF.loc[_name][['left_eye_x', 'left_eye_y']].astype(np.float64))
        right_eye = list(outputDF.loc[_name][['right_eye_x', 'right_eye_y']].astype(np.float64))
        nose = list(outputDF.loc[_name][['nose_x', 'nose_y']].astype(np.float64))

        # 
        # read image
        img = cv2.imread(_img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # cut the face
    
        facial_img = img[bbox[1]:bbox[3], bbox[0]: bbox[2]]
        if facial_img.size == 0:
            outputDF.loc[_name]['faces_not_found'] = 1
            outputDF.loc[_name]['faces_number'] = 0
            continue

        # align it 
        aligned_facial_img = postprocess.alignment_procedure(facial_img, right_eye, left_eye)#, nose)

        # face mesh prediction
        img_facemesh = aligned_facial_img.copy()
        resp = extract_eye_region(face=img_facemesh, facemeshnet=facemeshnet, iris_net=iris_net)
        
        # eye_region, eye_corners, pupil_center, eye = extract_eye_region(face=face, facemeshnet=facemeshnet, iris_net=iris_net)
        # img_facemesh = cv2.resize(img_facemesh, (192, 192))
        # detections = facemeshnet.predict_on_image(img_facemesh).cpu().numpy()


        # # # # # # # # # # # # # # #
        # Analysis of color change  #
        # # # # # # # # # # # # # # #
        
        for _eye in resp:
            line_points, _std, _mean, _eyelids_dist = color_analysis(**resp[_eye])
            resp[_eye]['line_points'] = line_points
            resp[_eye]['std'] = _std
            resp[_eye]['mean_color'] = _mean
            resp[_eye]['eyelids_dist'] = _eyelids_dist
            
            outputDF.loc[_name][f'eyelids_dist_{_eye}'] = resp[_eye]['eyelids_dist']
            outputDF.loc[_name][f'mean_color_{_eye}'] = resp[_eye]['mean_color']
            outputDF.loc[_name][f'std_{_eye}'] = resp[_eye]['std']
            outputDF.loc[_name][f'line_points_{_eye}'] = resp[_eye]['line_points']

            # if _save_eye_change:
            #     saveEyeChange(eye_region=resp[_eye]['eye_region'], color_value=zi, file_name=color_change_image_name, line_points=line_points, line_type="pupil-pass")

        if _save_faces:
            _face_file_name = os.path.join(save_faces, img_name)
            cv2.imwrite(_face_file_name, cv2.cvtColor(aligned_facial_img, cv2.COLOR_RGB2BGR))

        
        # _eyelids_dists.append(_eyelids_dist)
        # _stds.append(_std)
        # _means.append(_mean)

            
        # straight between corners
        # _std = np.std(zi_straight)
        # _mean = np.mean(zi)
        # saveEyeChange(eye_region=eye_region, color_value=zi_straight, file_name=color_change_image_name, line_points=line_points)
    
    # processed_images = set(_images[start:end]).difference(frames_exception)
    return outputDF, processed_images, _frames_names#, _means, _stds, _eyelids_dists


if __name__=='__main__':
    args = parser()
    participant_id = args.participant_id
    start, end = args.range
    resume = args.resume
    _save_faces = args.save_faces

    frames_path=os.path.join(dataset_root, "BlinkingValidationSetVideos",participant_id, "frames") 

    # Input
    input_path = os.path.join(dataset_root, 'tracked_faces', f'{participant_id}')
    input_path_file_hdf5 = os.path.join(input_path, "faceinfo.hdf5")

    # Output
    output_path = os.path.join(dataset_root, "eye_landmarks", f"{participant_id}")
    output_file_path_csv = os.path.join(output_path, f"eyeinfo.csv")
    output_file_path_hdf5 = os.path.join(output_path, f"eyeinfo.hdf5")

    # checking
    assert os.path.exists(frames_path), f"frames folder {frames_path} not found"
    assert os.path.exists(input_path_file_hdf5), f"faces info file {input_path_file_hdf5} not found"
    # 
    



    # input
    with pd.HDFStore(input_path_file_hdf5) as store:
        _data_df = store['tracked_faces_dataset_01']

    # except frames
    _except_frames = []
    if os.path.exists(output_file_path_hdf5) and resume and os.path.exists(output_file_path_hdf5):
        with pd.HDFStore(output_file_path_hdf5) as store:
            _data_df = store['eyes_info_dataset_01']
            # _except_frames = store['except_frames']
            _except_frames.extend(list(_data_df.loc[participant_id].index))
    
    else:
        
        shutil.rmtree(output_path, ignore_errors=True)
        os.makedirs(output_path, exist_ok=True)


    # load images
    _images = sorted(glob.glob(f"{frames_path}/*.png")) 
    if end == -1: end = len(glob.glob(f"{frames_path}/*.png"))

    # Load models:
    # face mesh
    facemeshnet = FaceMesh().to(gpu)
    facemeshnet.load_weights(FACE_MESH_MODEL_PATH)
    iris_net = IrisLandmarks().to(gpu)
    iris_net.load_weights(IRIS_MODEL_PATH)


    total_range = end-start
    batch = args.batch
    _iterations = ceil(total_range/batch)

    _data_df = _data_df.loc[(_data_df.index.get_level_values('participant_id') == participant_id) & (_data_df.index.get_level_values('face_id') == 'face_1')].reset_index(level=['participant_id', 'face_id'])
    # outputDF = _data_df.copy()


    for i in tqdm.tqdm(range(_iterations), total=_iterations):
        _batch_start = start+batch*i
        _batch_end = min(start+batch*(i+1),end)
        print(f"batch {i} --> {_batch_start}, {_batch_end}")
        # get images
        images_names = [os.path.basename(_path).split(".")[0] for _path in _images[_batch_start: _batch_end]]
        currentInputDF = _data_df.loc[_data_df.index.isin(images_names)]
        
        currentOutputDF, processed_images, _ = predict_eye_region(
                                                        images_paths=_images[_batch_start: _batch_end], 
                                                        facesInfo=currentInputDF, 
                                                        facemeshnet=facemeshnet, 
                                                        iris_net=iris_net, 
                                                        frames_exception=_except_frames, 
                                                        save_faces=_save_faces, 
                                                        save_eye_change="")

        # reindexing
        currentOutputDF = currentOutputDF.set_index(keys=['participant_id', 'face_id'], append=True).reorder_levels([1,0,2])
        if os.path.exists(output_file_path_hdf5):
            with pd.HDFStore(output_file_path_hdf5) as store:
                stored_data_df = store['eyes_info_dataset_01']
                metadata = store.get_storer('eyes_info_dataset_01').attrs.metadata
            concatenated_df = pd.concat([stored_data_df, currentOutputDF])
        else:
            concatenated_df = currentOutputDF.copy()

        
        
        concatenated_df = concatenated_df[~concatenated_df.index.duplicated(keep='last')]
        concatenated_df = concatenated_df.sort_index()
        # save to csv
        concatenated_df.to_csv(output_file_path_csv)
        # save
        # save to hd5
        store = pd.HDFStore(output_file_path_hdf5)
        store.put('eyes_info_dataset_01', concatenated_df)
        # store.put('except_frames', _frames_names)
        metadata = {
            'info':"""
                    color values are obtained alongsidethe 
                    lines segments connecting the center of the iris 
                    with the two corners of the eye as a series starting from the left corner of the eye,\n
                    the following informations are available:
                    mean color value\n
                    standard deviation\n
                    distance between the upper and the lower eyelids `directed_hausdorff`
                    """
            }
        store.get_storer('eyes_info_dataset_01').attrs.metadata = metadata
        store.close()
        print(f"results saved into {output_file_path_hdf5}")