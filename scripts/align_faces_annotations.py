import argparse
import os 
import numpy as np
import pickle
import tqdm
from blinkdetect.common import read_bbox_tag

dataset_root = os.path.join(os.path.dirname(__file__), "..", "dataset")

def parser():
    _parser = argparse.ArgumentParser()
    _parser.add_argument('--dataset', required=True, choices=["BlinkingValidationSetVideos", "eyeblink8", "talkingFace", "zju", "RN"])
    return _parser.parse_args()

def extract_faces(annotations, all_detections):
    """
    """
    _detections = {}

    for _name in tqdm.tqdm(sorted(all_detections.keys()), total=len(all_detections.keys()), leave=False, desc="frame"):
        # annotation box
        org_bbox = annotations[_name]
        org_bbox = [int(i) for i in org_bbox]
        org_center = np.array([(org_bbox[0]+0.5*org_bbox[2]), (org_bbox[1]+0.5*org_bbox[3])])

        # face detected
        dets = all_detections[_name]

        min_norm = 10000
        final_dets = {'face_1': {}}
        for face_id in dets:
            bbox_tmp = dets[face_id]['facial_area']
            center_tmp = np.array([(bbox_tmp[0]+bbox_tmp[2])/2, (bbox_tmp[1]+bbox_tmp[3])/2])
            _norm = np.linalg.norm(center_tmp - org_center)
            if _norm < min_norm:
                final_dets['face_1'] = dets[face_id]
        
        _detections[_name]={
            "faces": final_dets, 
            "faces_not_found": all_detections[_name]['faces_not_found'], 
            "faces_number": 1
        }
        
    return _detections



if __name__=="__main__":

    args = parser()
    dataset = os.path.normpath(os.path.join(dataset_root, args.dataset))

    #
    # video paths
    videos_paths = []
    for root,dirs, files in os.walk(dataset):
        for dir in files:
            name, ext = os.path.splitext(dir)
            if ext in [".tag"]:
                videos_paths.append(os.path.join(root,dir))
    #
    # 
    videos_progress = tqdm.tqdm(videos_paths, total=len(videos_paths), desc="face detection")
    for video_path in videos_progress:
        # input
        
        video_name = os.path.dirname(video_path)
        video_name = os.path.relpath(video_name, dataset)
        videos_progress.set_postfix(video=video_name)

        # output
        faces_root = os.path.normpath(os.path.join(dataset_root,"faces", args.dataset, video_name))
        faceinfo_file_path_pkl = os.path.normpath(os.path.join(faces_root, "faceinfo_v2.pkl"))
        assert os.path.exists(faceinfo_file_path_pkl), f"faceinfo file is does not exist {faceinfo_file_path_pkl}"

        # read annotations and estimations
        # estimations
        with open( faceinfo_file_path_pkl, "rb" ) as pkl_file:
            all_detections = pickle.load(pkl_file)
        # annotations
        annotations = read_bbox_tag(video_path)
        
        # alignment
        _new_detections = extract_faces(annotations=annotations, all_detections=all_detections)

        
        # save the results
        with open( faceinfo_file_path_pkl, "wb" ) as pkl_file:
            pickle.dump(_new_detections, pkl_file)
