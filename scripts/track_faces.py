import argparse
import os 
import glob
from math import ceil
import sys
if sys.version_info <= (3, 7):
    import pickle5 as pickle
else:
    import pickle
import tqdm
import shutil
from deepface.DeepFace import build_model, represent
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from deepface.commons import functions,  distance as dst
import time
import sys
lib_dir = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(lib_dir)
# 
from blinkdetect.tracking.klt_tracker import KLT
# 
from scipy.spatial.distance import directed_hausdorff
import scipy.optimize


dataset_root = os.path.join(os.path.dirname(__file__), "..", "dataset")

def parser():
    _parser = argparse.ArgumentParser()
    _parser.add_argument('--dataset', required=True, choices=["BlinkingValidationSetVideos", "eyeblink8", "talkingFace", "zju", "RN"])
    _parser.add_argument('-rng', '--range', type=int, default=[0,-1], nargs=2)
    _parser.add_argument('--batch', type=int, default=32, help='number of frames to be saved as a batch')
    _parser.add_argument('--resume', action='store_true', help='if true existed frames of an existed participant will not be replaced')
    # _parser.add_argument('--deepface', action='store_true', help='if to use deepface in tracking or not')
    _parser.add_argument('--method', type=str, choices=['deepface', 'klt', 'normal'], default='normal')
    _parser.add_argument('--limit', type=int, default=40)
    _parser.add_argument('--closest', action='store_true', help="wheather to consider the closest pairs of feature points between the detections and the tracked points")
    # deepface=True

    return _parser.parse_args()

model_name = 'VGG-Face'
model = build_model(model_name)

def verify(img1_path, img2_path = '', model_name = 'VGG-Face', distance_metric = 'cosine', model = None, enforce_detection = True, detector_backend = 'opencv', align = True, prog_bar = True):

	"""
	This function verifies an image pair is same person or different persons.
	Parameters:
		img1_path, img2_path: exact image path, numpy array or based64 encoded images could be passed. If you are going to call verify function for a list of image pairs, then you should pass an array instead of calling the function in for loops.
		e.g. img1_path = [
			['img1.jpg', 'img2.jpg'],
			['img2.jpg', 'img3.jpg']
		]
		model_name (string): VGG-Face, Facenet, OpenFace, DeepFace, DeepID, Dlib, ArcFace or Ensemble
		distance_metric (string): cosine, euclidean, euclidean_l2
		model: Built deepface model. A face recognition model is built every call of verify function. You can pass pre-built face recognition model optionally if you will call verify function several times.
			model = DeepFace.build_model('VGG-Face')
		enforce_detection (boolean): If any face could not be detected in an image, then verify function will return exception. Set this to False not to have this exception. This might be convenient for low resolution images.
		detector_backend (string): set face detector backend as retinaface, mtcnn, opencv, ssd or dlib
		prog_bar (boolean): enable/disable a progress bar
	Returns:
		Verify function returns a dictionary. If img1_path is a list of image pairs, then the function will return list of dictionary.
		{
			"verified": True
			, "distance": 0.2563
			, "max_threshold_to_verify": 0.40
			, "model": "VGG-Face"
			, "similarity_metric": "cosine"
		}
	"""

	tic = time.time()

	img_list, bulkProcess = functions.initialize_input(img1_path, img2_path)

	resp_objects = []

	#--------------------------------

	model_names = []; metrics = []
	model_names.append(model_name)
	metrics.append(distance_metric)

	#--------------------------------

	models = {}
	models[model_name] = model

	#------------------------------

	for index in range(0,len(img_list)):

		instance = img_list[index]

		if type(instance) == list and len(instance) >= 2:
			img1_path = instance[0]; img2_path = instance[1]

			ensemble_features = []

			for i in  model_names:
				custom_model = models[i]

				#img_path, model_name = 'VGG-Face', model = None, enforce_detection = True, detector_backend = 'mtcnn'
				img1_representation = represent(img_path = img1_path
						, model_name = model_name, model = custom_model
						, enforce_detection = enforce_detection, detector_backend = detector_backend
						, align = align)

				img2_representation = represent(img_path = img2_path
						, model_name = model_name, model = custom_model
						, enforce_detection = enforce_detection, detector_backend = detector_backend
						, align = align)

				#----------------------
				#find distances between embeddings

				for j in metrics:

					if j == 'cosine':
						distance = dst.findCosineDistance(img1_representation, img2_representation)
					elif j == 'euclidean':
						distance = dst.findEuclideanDistance(img1_representation, img2_representation)
					elif j == 'euclidean_l2':
						distance = dst.findEuclideanDistance(dst.l2_normalize(img1_representation), dst.l2_normalize(img2_representation))
					else:
						raise ValueError("Invalid distance_metric passed - ", distance_metric)

					distance = np.float64(distance) #causes trobule for euclideans in api calls if this is not set (issue #175)
					#----------------------
					#decision
					if model_name != 'Ensemble':

						threshold = dst.findThreshold(i, j)

						if distance <= threshold:
							identified = True
						else:
							identified = False

						resp_obj = {
							"verified": identified
							, "distance": distance
							, "max_threshold_to_verify": threshold
							, "model": model_name
							, "similarity_metric": distance_metric

						}
						if bulkProcess == True:
							resp_objects.append(resp_obj)
						else:
							return resp_obj

			#----------------------

		else:
			raise ValueError("Invalid arguments passed to verify function: ", instance)

	#-------------------------

	return resp_objects

	if bulkProcess == True:

		resp_obj = {}

		for i in range(0, len(resp_objects)):
			resp_item = resp_objects[i]
			resp_obj["pair_%d" % (i+1)] = resp_item

		return resp_obj

# Frame: N
# track the positions of the faces in frame N-1 using KLT: tracked_faces
# for each tracked_face in tracked_faces:
    # for each detected face in the current frame:
        # compute the distance to tracked_face
    # get the bbox with the samllest distance: dist, closest_face
    # if dist is less greater than a threshold:
        # compute the similarity between the tracked face and the closest face
        # if they are distant from each other: the closest_face is a new face
    
    # else: -- update the tracking information and assign the tracked_face_id to the closest_face



        # if the status is 0
            # update with location of the closest within limit bounding box
            # the distance is computed in different ways:
                # the average of the distances between the feature points
                # hausdorff distance
                # the distance between the centers of the two sets of feature points
        # else:
            # update with the new tracked points
            # suggestion:
                # if the distance is low enough update with the feature points from the face detector
        # assign the face id
# Another idea:
    # form an energy function


def symmetric_hausdorff(A: np.ndarray, B: np.ndarray):
    _A = A.reshape((-1,2))
    _B = B.reshape((-1,2))
    return max(directed_hausdorff(_A, _B)[0], directed_hausdorff(_B, _A)[0])

def closest_pairs(A: np.ndarray, B: np.ndarray):
    """[summary]
    https://stackoverflow.com/questions/65373827/fastest-way-to-find-the-nearest-pairs-between-two-numpy-arrays-without-duplicate

    Args:
        A (np.ndarray): the array with the higher number of rows
        B (np.ndarray): the array with the lower or equal number of rows

    Returns:
        (np.ndarray): the rows in A that are the closest to B
    """
    cost = np.linalg.norm(B.reshape(-1,2)[:,np.newaxis,:] - A.reshape(-1,2), axis=2)
    _, indexes = scipy.optimize.linear_sum_assignment(cost)
    return A[indexes]

def track_faces_v1(img_path, dets, prev_faces={},frame_number="", closest = False):

    current_faces = {}
    prev_faces_keys = sorted(prev_faces.keys())

    #--------------------------- same 

    if (type(img_path) == str and img_path != None): #exact image path

        if os.path.isfile(img_path) != True:
            raise ValueError("Confirm that ",img_path," exists")

        img = cv2.imread(img_path)

    if (isinstance(img_path, np.ndarray) and img_path.any()): #numpy array
        img = img_path.copy()

    #---------------------------

    klt_tracker = KLT()

    #---------------------------

    
    _face_idx = 1

    #--------------------------- track previous faces
    if len(prev_faces) > 0:
        for face in prev_faces_keys:
            prev_img = cv2.imread(prev_faces[face]['_path'])
            # 
            klt_tracker.setOldframe(prev_img)

            klt_tracker.setFeatures(prev_faces[face]['tracking_info']['feature_points'])
            klt_resp, status = klt_tracker.track(img)
            
            # update
            prev_faces[face]['tracking_info']['status'] = status
            if status:
                prev_faces[face]['tracking_info']['feature_points'] = klt_resp['good_new'].reshape(-1,1,2)
        
    #---------------------------
    # for each detected faces in the current frame:
    
    if type(dets) == dict:
        dets_keys = sorted(dets.keys())
        assigned = []
        if len(prev_faces) > 0:
            # exit()
            prev_faces_keys = sorted(prev_faces.keys())
            for face in prev_faces_keys:
                tracked_feature_points = prev_faces[face]['tracking_info']['feature_points'].copy()
                _dists = []
                for key in dets_keys:
                    if key in assigned: 
                        continue
                    # 
                    identity = dets[key]
                    # For alignment
                    landmarks = identity["landmarks"]
                    feature_points = np.array([[landmarks["left_eye"]], [landmarks["right_eye"]], [landmarks["nose"]], [landmarks["mouth_right"]], [landmarks['mouth_left']]])
                    if feature_points.shape[0] != tracked_feature_points.shape[0] and closest:
                        feature_points = closest_pairs(feature_points, tracked_feature_points)
                    # compute distance
                    _dist = np.linalg.norm(np.mean(tracked_feature_points, axis=0) - np.mean(feature_points, axis=0))
                    _dists.append({"face_id": key, "distance": _dist})
                
                if len(_dists) == 0: continue
                
                # find minimum
                _bbox_idx = min(enumerate(_dists), key=lambda x: x[1]['distance'])[0]
                # if satisfied assign the bbox to this face

                det_id = _dists[_bbox_idx]['face_id']
                assigned.append(det_id)
                
                
                # the closest
                identity = dets[det_id]
                facial_area = identity["facial_area"]
                facial_img = img[facial_area[1]: facial_area[3], facial_area[0]: facial_area[2]]
                landmarks = identity["landmarks"]
                left_eye = landmarks["left_eye"]
                right_eye = landmarks["right_eye"]
                nose = landmarks["nose"]
                _center = [(facial_area[1] + facial_area[3])/2, (facial_area[0]+ facial_area[2])/2]

                feature_points = np.array([[left_eye], [right_eye], [nose], [landmarks["mouth_right"]], [landmarks['mouth_left']]])
                
                if _dists[_bbox_idx]['distance'] > 5* abs(int(frame_number)-int(prev_faces[face]['frame'])):
                    resp = verify(img1_path=[
                        [prev_faces[face]["img_path"], facial_img]], 
                        enforce_detection=False, model=model)
                    
                    if resp[0]["verified"] and resp[0]['distance'] < 0.15:
                        matching_face_idx = face
                    else:
                        # prev_faces
                        matching_face_idx = f"face_{len(prev_faces)}"
                else:
                    matching_face_idx = face

                current_faces[matching_face_idx] = {
                    '_path':img_path,
                    'frame': frame_number,
                    'img_path': facial_img,#[:, :, ::-1],
                    'facial_area': {"bbox": facial_area, "mode": "XYXY"},
                    'left_eye': left_eye,
                    'right_eye': right_eye,
                    'nose': nose,
                    'center': np.array(_center),
                    'tracking_info': {'feature_points': feature_points.copy()},
                    'yaw': dets[det_id]['yaw'],
                    'pitch': dets[det_id]['pitch'],
                    'roll': dets[det_id]['roll']
                }

        # no previous faces were detected
        else:
            for key in dets_keys:
                matching_face_idx = f"face_{_face_idx}"
                _face_idx = _face_idx + 1

                identity = dets[key]
                facial_area = identity["facial_area"]
                facial_img = img[facial_area[1]: facial_area[3], facial_area[0]: facial_area[2]]
                # For alignment
                landmarks = identity["landmarks"]
                left_eye = landmarks["left_eye"]
                right_eye = landmarks["right_eye"]
                nose = landmarks["nose"]
                _center = [(facial_area[1] + facial_area[3])/2, (facial_area[0]+ facial_area[2])/2]

                feature_points = np.array([[landmarks["left_eye"]], [landmarks["right_eye"]], [landmarks["nose"]], [landmarks["mouth_right"]], [landmarks['mouth_left']]])
                #
                current_faces[matching_face_idx] = {
                    '_path':img_path,
                    'frame': frame_number,
                    'img_path': facial_img,
                    'facial_area': {"bbox": facial_area, "mode": "XYXY"},
                    'left_eye': left_eye,
                    'right_eye': right_eye,
                    'nose': nose,
                    'center': np.array(_center),
                    'tracking_info': {"feature_points": feature_points.copy()},
                    'yaw': dets[key]['yaw'],
                    'pitch': dets[key]['pitch'],
                    'roll': dets[key]['roll']
                }

        # for each tracked object:
            # if the status is 0
                # update with location of the closest within limit bounding box
                # the distance is computed in different ways:
                    # the average of the distances between the feature points
                    # hausdorff distance
                    # the distance between the centers of the two sets of feature points
            # else:
                # update with the new tracked points
                # suggestion:
                    # if the distance is low enough update with the feature points from the face detector
            # assign the face id
        
    return current_faces

def track_faces(img_path, dets, prev_faces={},frame_number="", method='deepface', align = True, limit=40):
    """return {} if faces are not found

    Args:
        img_path ([type]): [description]
        dets ([type]): [description]
        prev_faces (dict, optional): [description]. Defaults to {}.
        align (bool, optional): [description]. Defaults to True.

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """
    # resp = {}
    current_faces = {}
    prev_faces_keys = sorted(prev_faces.keys())

    #---------------------------

    if (type(img_path) == str and img_path != None): #exact image path

        if os.path.isfile(img_path) != True:
            raise ValueError("Confirm that ",img_path," exists")

        img = cv2.imread(img_path)

    if (isinstance(img_path, np.ndarray) and img_path.any()): #numpy array
        img = img_path.copy()

    #---------------------------

    if method == 'klt':
        klt_tracker = KLT()

    #---------------------------

    obj = dets#detect_faces(img_path = img, threshold = threshold, model = model)
    _face_idx = 1

    if type(obj) == dict:

        for key in obj:
            prev_faces_keys = sorted(prev_faces.keys())
            identity = obj[key]

            facial_area = identity["facial_area"]
            facial_img = img[facial_area[1]: facial_area[3], facial_area[0]: facial_area[2]]
            # left, right, top, down
            # For alignment
            landmarks = identity["landmarks"]
            left_eye = landmarks["left_eye"]
            right_eye = landmarks["right_eye"]
            nose = landmarks["nose"]
            _center = [(facial_area[1] + facial_area[3])/2, (facial_area[0]+ facial_area[2])/2]

            # if align == True:
            #     facial_img = postprocess.alignment_procedure(facial_img, right_eye, left_eye, nose)
            if obj[key]['yaw'] > limit or obj[key]['yaw'] < -limit:
                continue
            

            if len(prev_faces) > 0:
                if method =='deepface':
                    resp = verify(img1_path=[
                        [prev_faces[face]["img_path"], facial_img] for face in prev_faces_keys
                        ], 
                        enforce_detection=False, model=model)
                    
                    verified_pairs = [
                        {
                            "distance": _item['distance'], 'face': prev_faces_keys[idx]
                        } 
                        for idx, _item in enumerate(resp) 
                        if (_item["verified"] and _item['distance'] < 0.15)]# or len(prev_faces)==2]
                    
                    if len(verified_pairs) > 0:
                        _face_idx = min(enumerate(verified_pairs), key=lambda x: x[1]['distance'])[0]
                        matching_face_idx = verified_pairs[_face_idx]['face']
                    
                    else:
                        # add new face
                        # matching_face_idx = f"face_{len(prev_faces_keys)+1}"
                        return current_faces
                        # append it to the list of previous detected faces
                        # prev_faces[matching_face_idx] = {"img_path"}
                        # prev_faces_keys.append(matching_face_idx)

                elif method == 'klt':
                    resp = verify(img1_path=[
                        [prev_faces[face]["img_path"], facial_img] for face in prev_faces_keys
                        ], 
                        enforce_detection=False, model=model)
                    
                    _norms = []
                    for face in prev_faces_keys:
                        prev_img = cv2.imread(prev_faces[face]['_path'])
                        klt_tracker.setOldframe(prev_img)
                        klt_tracker.setFeatures(prev_faces[face]['feature_points'])
                        klt_resp, status = klt_tracker.track(img)
                        # check status
                        if status:
                            tracked_new_points = klt_resp['good_new']
                            tracked_center = np.mean(tracked_new_points, axis=0)
                            # 
                            _norms.append(np.linalg.norm(tracked_center - _center))
                        # 
                        else:
                            # TODO
                            print()


                else:
                    
                    _norms = [np.linalg.norm(np.array(prev_faces[face]["center"]) - np.array(_center)) for face in prev_faces_keys]

                    verified_pairs = [
                        {
                            "distance": _item, 'face': prev_faces_keys[idx]
                        } 
                        for idx, _item in enumerate(_norms) 
                        if _item < 10 * (int(frame_number)-int(prev_faces[prev_faces_keys[idx]]['frame']))
                        ]
                    # min norm
                    if len(verified_pairs) > 0:
                        _face_idx = min(enumerate(verified_pairs), key=lambda x: x[1]['distance'])[0]
                        matching_face_idx = verified_pairs[_face_idx]['face']
                    else:
                        #  matching_face_idx = f"face_{len(prev_faces_keys)+1}"
                        return current_faces
            
            else:
                # no previous faces were detected
                matching_face_idx = f"face_{_face_idx}"
                _face_idx = _face_idx + 1
            
            if method == 'klt': feature_points = np.array([[left_eye], [right_eye], [nose], [landmarks["mouth_right"]], [landmarks['mouth_left']]])
            else: feature_points = []
            current_faces[matching_face_idx] = {
                '_path':img_path,
                'frame': frame_number,
                'img_path': facial_img,#[:, :, ::-1],
                'facial_area': {"bbox": facial_area, "mode": "XYXY"},
                'left_eye': left_eye,
                'right_eye': right_eye,
                'nose': nose,
                'center': np.array(_center),
                'feature_points': feature_points
                }


    return current_faces

def track_faces_batch(
    images_paths: list, 
    detections, 
    start: int = 0, end: int = -1, 
    frames_exception: list=[], 
    prev_faces={}, 
    method='deepface', 
    limit_angle = 40,
    closest = False):
    """[summary]

    Args:
        images_paths (list): [description]
        detections ([type]): [description]
        start (int, optional): [description]. Defaults to 0.
        end (int, optional): [description]. Defaults to -1.
        frames_exception (list, optional): [description]. Defaults to [].
        prev_faces (dict, optional): [description]. Defaults to {}.
        method (str, optional): [description]. Defaults to 'deepface'.
        limit_angle (int, optional): [description]. Defaults to 40.
        closest (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    # ---------------
    _all_faces = {**prev_faces}
    # ---------------
    _detections = {}

    _images = images_paths[start:end]

    for _img_path in tqdm.tqdm(_images, total=len(_images), leave=False, desc="frame"):
        # 
        img_name = os.path.basename(_img_path)
        _name, _ext = img_name.split(".")

        # check if it is not intended to be processed
        if _name in frames_exception:
            continue

        # add path and frames number

        # face tracking
        _current_faces = track_faces_v1(img_path = _img_path, dets=detections[_name]['faces'], prev_faces=_all_faces, frame_number=_name, closest=closest)
        # TODO: best way to merge two dics
        _all_faces = {**_all_faces, **_current_faces}
        # 
        if len(_current_faces) > 0 :
            faces_not_found=0
            n_faces = len(_current_faces)
        else:
            faces_not_found=1
            n_faces = 0
        # 
        _detections[_name]={
            "img_path": _img_path,
            "faces": _current_faces, 
            "faces_not_found": faces_not_found, 
            "faces_number": n_faces
        }
          
    return _detections, _all_faces

if __name__=="__main__":


    args = parser()
    start, end = args.range
    resume = args.resume
    method = args.method
    _limit=args.limit
    _closest = args.closest
    dataset = os.path.join(dataset_root, args.dataset)

    # videos paths
    videos_paths = []
    for root,dirs, files in os.walk(dataset):
        for dir in files:
            name, ext = os.path.splitext(dir)
            if ext in [".avi", ".mov", ".wmv", ".mp4"]:
                videos_paths.append(os.path.join(root,dir))
    # 
    videos_progress = tqdm.tqdm(videos_paths, total=len(videos_paths), desc="vid")
    for video_path in videos_progress:
        # input
        video_name = os.path.dirname(video_path)
        video_name = os.path.relpath(video_name, dataset)
        # 
        videos_progress.set_postfix(video=video_name)
        # 
        frames_root = os.path.normpath(os.path.join(os.path.dirname(video_path), "frames"))
        if not os.path.exists(frames_root):
            continue
        faces_detection_file_path = os.path.normpath(os.path.join(dataset_root,"faces", args.dataset, video_name, 'faceinfo_v2.pkl'))
        if not os.path.exists(faces_detection_file_path):
            continue

        # output
        tracked_faces_root = os.path.join(dataset_root,"tracked_faces", args.dataset, video_name)
        faceinfo_file_path_csv = os.path.normpath(os.path.join(tracked_faces_root, "faceinfo.csv"))
        faceinfo_file_path_hdf5 = os.path.normpath(os.path.join(tracked_faces_root, "faceinfo.hdf5"))
        last_faces_info = os.path.normpath(os.path.join(tracked_faces_root, 'last_faces'))

        # emptying the folder in case of not resuming
        if not resume and os.path.exists(tracked_faces_root):
            shutil.rmtree(tracked_faces_root)

        os.makedirs(tracked_faces_root,exist_ok=True)
        os.makedirs(last_faces_info,exist_ok=True)

        # when resume is set, existed participant_id,frame_num indices will not be processed
        _except_frames = [] # for resuming
        _last_faces = {} # for tracking

        _data_df=None
        if resume:
            if os.path.exists(last_faces_info):
                _faces_paths = glob.glob(f"{last_faces_info}/*.jpg")
                for _face_path in _faces_paths:
                    # 
                    _face = cv2.imread(_face_path)
                    img_name = os.path.basename(_face_path)
                    _face_id, _ext = img_name.split(".")
                    _last_faces[_face_id] = {"img_path": _face}
                
                if os.path.exists(faceinfo_file_path_hdf5):
                    with pd.HDFStore(faceinfo_file_path_hdf5) as store:
                        _data_df = store['tracked_faces_dataset_01']

                    if video_name in _data_df.index:
                        _except_frames.extend(list(_data_df.loc[video_name].index))
    
        # load images
        _images = sorted(glob.glob(f"{frames_root}/*.png")) 
        # if end == -1: 
        start = 0
        end = len(glob.glob(f"{frames_root}/*.png"))

        # load detections
        with open(faces_detection_file_path, "rb") as _dets_file:
            _detections = pickle.load(_dets_file)

        total_range = end-start
        batch = args.batch
        _iterations = ceil(total_range/batch)
        # 
        iterations_progress = tqdm.tqdm(range(_iterations), total=_iterations, leave=False, desc="batch")
        for i in iterations_progress:
            _batch_start = start+batch*i
            _batch_end = min(start+batch*(i+1),end)
            # 
            _tracking_resp, _last_faces = track_faces_batch(images_paths=_images, detections=_detections,start=_batch_start, end=_batch_end, frames_exception=_except_frames, prev_faces=_last_faces, method=method, limit_angle=_limit, closest=_closest)
            
            # Save last faces:
            for _face in _last_faces:
                _path = os.path.join(last_faces_info, f"{_face}.jpg")
                cv2.imwrite(_path, _last_faces[_face]['img_path'])

            # save the results
            # if os.path.exists(eyeinfo_file_path_hdf5):
            #     with pd.HDFStore(eyeinfo_file_path_hdf5) as store:
            #         _data_df = store['dataset_02']
            #         metadata = store.get_storer('dataset_02').attrs.metadata
            
            data_array = []
            for _frame in _tracking_resp:
                # _tracking_resp[_frame]
                _faces_not_found = _tracking_resp[_frame]['faces_not_found']
                _faces_number = _tracking_resp[_frame]['faces_number']
                _img_path = _tracking_resp[_frame]['img_path']
                left,top,right,bottom = [0,0,0,0]
                left_eye_x, left_eye_y = [0,0]
                right_eye_x, right_eye_y = [0,0]
                nose_x, nose_y = [0,0]
                yaw, pitch, roll = [0,0,0]
                if _faces_number == 0:
                    data_array.append([video_name, _frame, _face,_img_path, _faces_not_found, _faces_number, left,top,right,bottom, left_eye_x, left_eye_y, right_eye_x, right_eye_y, nose_x, nose_y, yaw, pitch, roll])
                # face data
                for _face in _tracking_resp[_frame]['faces']:
                    left,top,right,bottom = _tracking_resp[_frame]['faces'][_face]['facial_area']['bbox']
                    left_eye_x, left_eye_y = _tracking_resp[_frame]['faces'][_face]['left_eye']
                    right_eye_x, right_eye_y = _tracking_resp[_frame]['faces'][_face]['right_eye']
                    nose_x, nose_y = _tracking_resp[_frame]['faces'][_face]['nose']
                    # 
                    yaw = _tracking_resp[_frame]['faces'][_face]['yaw']
                    pitch = _tracking_resp[_frame]['faces'][_face]['pitch']
                    roll = _tracking_resp[_frame]['faces'][_face]['roll']

                    data_array.append([video_name, _frame, _face,_img_path, _faces_not_found, _faces_number, left,top,right,bottom, left_eye_x, left_eye_y, right_eye_x, right_eye_y, nose_x, nose_y, yaw, pitch, roll])
            
            # 
            # INDEX
            # 
            data_np = np.array(data_array)
            arrays = [
                data_np[:,0],
                data_np[:,1],
                data_np[:,2]
                ]

            tuples_indices = list(zip(*arrays))
            index = pd.MultiIndex.from_tuples(tuples_indices, names=["participant_id", "frame_num", "face_id"])

            # 
            # initialize dataframe
            # 
            # _data_df = pd.DataFrame(
            #     columns=["path", "mean_color", "std", "eyelids_dist"], 
            #     index=index, )
            # _data_df.astype(dtype={
            #         # "participant_id":str, 
            #         # "frame_num":str, 
            #         "path":str, 
            #         "mean_color":np.float64, 
            #         "std":np.float64, 
            #         "eyelids_dist":np.float64})

            if os.path.exists(faceinfo_file_path_hdf5):
                with pd.HDFStore(faceinfo_file_path_hdf5) as store:
                    _data_df = store['tracked_faces_dataset_01']
                    metadata = store.get_storer('tracked_faces_dataset_01').attrs.metadata
            else:
                # 
                # initialize dataframe
                # 
                _data_df = pd.DataFrame(
                    columns=['img_path', 'faces_not_found', 'faces_number', 'left','top','right','bottom', 'left_eye_x', 'left_eye_y', 'right_eye_x', 'right_eye_y', 'nose_x', 'nose_y', 'yaw', 'pitch', 'roll'], 
                    index=index, )
            

            new_df =  pd.DataFrame(
                data_np[:,3:], 
                index=index,
                columns=['img_path', 'faces_not_found', 'faces_number', 'left','top','right','bottom', 'left_eye_x', 'left_eye_y', 'right_eye_x', 'right_eye_y', 'nose_x', 'nose_y', 'yaw', 'pitch', 'roll'])
            
            # merge the two dataframes
            concatenated_df = pd.concat([_data_df, new_df])
            concatenated_df = concatenated_df[~concatenated_df.index.duplicated(keep='last')]
            concatenated_df = concatenated_df.sort_index()

            # save to csv
            concatenated_df.to_csv(faceinfo_file_path_csv)

            # save to hd5
            store = pd.HDFStore(faceinfo_file_path_hdf5)
            store.put('tracked_faces_dataset_01', concatenated_df)      
            metadata = {
                'info':"""
                        using retina-face
                        """
                }
            store.get_storer('tracked_faces_dataset_01').attrs.metadata = metadata
            store.close()

            # iterations_progress.set_postfix(f"results saved into {video_name}")
            # print(f"results saved into {faceinfo_file_path_hdf5}")
        iterations_progress.close()
    videos_progress.close()
