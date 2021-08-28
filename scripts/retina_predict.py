import argparse
import os 
import glob
from math import ceil
import tqdm
from retinaface import RetinaFace
import numpy as np
import pandas as pd
import cv2
# from deepface import DeepFace

dataset_root = os.path.join(os.path.dirname(__file__), "..", "dataset")

def parser():
    _parser = argparse.ArgumentParser()
    _parser.add_argument('-pid', '--participant_id', required=True)
    _parser.add_argument('-rng', '--range', type=int, default=[0,-1], nargs=2)
    _parser.add_argument('--batch', type=int, default=32, help='number of frames to be saved as a batch')
    _parser.add_argument('--resume', action='store_true', help='if true existed frames of an existed participant will not be replaced')

    return _parser.parse_args()


retina_extract_faces = RetinaFace.extract_faces
model = RetinaFace.build_model()


def extract_faces(images_paths: list, start: int = 0, end: int = -1, frames_exception: list=[], output_dir: str = './output_faces'):
    """
    """

    os.makedirs(output_dir, exist_ok=True)
    # 
    _frames_paths = []
    _frames_names = []
    faces_not_found = []
    # last_images

    _images = images_paths

    for _img_path in tqdm.tqdm(_images[start:end], total=len(_images[start:end])):
        # 
        img_name = os.path.basename(_img_path)
        _name, _ext = img_name.split(".")

        # check if it is not intended to be processed
        if _name in frames_exception:
            continue

        # add path and frames number
        _frames_paths.append(_img_path)
        _frames_names.append(_name)

        # face detection
        faces = retina_extract_faces(img_path = _img_path, model=model, align = True)
        if len(faces) == 0:
            faces_not_found.append(1)
            continue

        # face = faces[0]
        faces_not_found.append(0)
        for _idx,face in enumerate(faces):
            # save image
            output_img = os.path.join(output_dir, f"{_name}_{_idx}.{_ext}")
            # print(output_img)
            cv2.imwrite(output_img, face)
        
        
    return faces_not_found, _frames_paths, _frames_names



if __name__=="__main__":

    args = parser()
    participant_id = args.participant_id
    start, end = args.range
    resume = args.resume

    # 
    frames_root=os.path.join(dataset_root, "BlinkingValidationSetVideos",participant_id, "frames")
    
    faces_root = os.path.join(dataset_root,"faces", participant_id)
    faceinfo_file_path_hdf5 = os.path.join(faces_root, "faceinfo.hdf5")
    faceinfo_file_path_csv = os.path.join(faces_root, "faceinfo.csv")

    os.makedirs(faces_root,exist_ok=True)

    # when resume is set, existed participant_id,frame_num indices will not be processed
    _except_frames = []
    if resume:
        if os.path.exists(faceinfo_file_path_hdf5):
            with pd.HDFStore(faceinfo_file_path_hdf5) as store:
                _data_df = store['faces_dataset_01']
                _except_frames.extend(list(_data_df.loc[participant_id].index))
                # print(_except_frames)

    
   # load images
    _images = sorted(glob.glob(f"{frames_root}/*.png")) 
    if end == -1: end = len(glob.glob(f"{frames_root}/*.png"))

    total_range = end-start
    batch = args.batch
    _iterations = ceil(total_range/batch)
    # print(_iterations)
    for i in tqdm.tqdm(range(_iterations), total=_iterations):
        _batch_start = start+batch*i
        _batch_end = min(start+batch*(i+1),end)
        faces_not_found, _frames_paths, _frames_names = extract_faces(
                                                            images_paths=_images, 
                                                            start=_batch_start, 
                                                            end=_batch_end, 
                                                            frames_exception=_except_frames, 
                                                            output_dir=faces_root)

        # save_results
        # 
        # INDEX
        # 
        # print(faces_not_found, _frames_paths, _frames_names)
        arrays = [
            [participant_id] * len(_frames_names),
            _frames_names
            ]
    
        tuples_indices = list(zip(*arrays))
        index = pd.MultiIndex.from_tuples(tuples_indices, names=["participant_id", "frame_num"])

        # 
        # initialize dataframe
        # 
        _data_df = pd.DataFrame(
            columns=["path", "face_not_found"], 
            index=index, )
        # _data_df.astype(dtype={
        #         "path":str, 
        #         "face_not_found": np.int32
        #         })

        if os.path.exists(faceinfo_file_path_hdf5):
            with pd.HDFStore(faceinfo_file_path_hdf5) as store:
                _data_df = store['faces_dataset_01']
                metadata = store.get_storer('faces_dataset_01').attrs.metadata
        
        new_df =  pd.DataFrame(
            {
                "path": _frames_paths, 
                "face_not_found": faces_not_found
            }, 
            index=index)

        # merge the two dataframes
        concatenated_df = pd.concat([_data_df, new_df])
        concatenated_df = concatenated_df[~concatenated_df.index.duplicated(keep='last')]
        concatenated_df = concatenated_df.sort_index()
        # save to csv
        concatenated_df.to_csv(faceinfo_file_path_csv)

        # save to hd5
        store = pd.HDFStore(faceinfo_file_path_hdf5)
        store.put('faces_dataset_01', concatenated_df)      
        metadata = {
            'info':"""
                    aligned faces extracted using retina-face 0.0.6: https://pypi.org/project/retinaface/
                    
                    """
            }
        store.get_storer('faces_dataset_01').attrs.metadata = metadata
        store.close()
        print(f"results saved into {faceinfo_file_path_hdf5}")
