import argparse
import os 
import glob

import tqdm
import shutil
import numpy as np
import pandas as pd
import cv2


def parser():
    _parser = argparse.ArgumentParser()
    _parser.add_argument('-pid', '--participant_id', required=True)
    _parser.add_argument('-rng', '--range', type=int, default=[0,-1], nargs=2)

    return _parser.parse_args()



if __name__=="__main__":

    args = parser()
    participant_id = args.participant_id
    start, end = args.range
    # resume = args.resume

    # 
    dataset_root = os.path.join(os.path.dirname(__file__), "..", "dataset")
    frames_root=os.path.join(dataset_root, "BlinkingValidationSetVideos",participant_id, "frames")
    tracked_faces_root = os.path.join(dataset_root,"tracked_faces", participant_id)
    faceinfo_file_path_hdf5 = os.path.join(tracked_faces_root, "faceinfo.hdf5")

    assert os.path.exists(faceinfo_file_path_hdf5), "tracking information is not available"

    tracked_faces_seq = os.path.join(tracked_faces_root, 'seq')
    if os.path.exists(tracked_faces_seq):
        shutil.rmtree(tracked_faces_seq)

    
    os.makedirs(tracked_faces_seq,exist_ok=True)

    # if resuming
    # _except_frames = []
    # _data_df=None

    with pd.HDFStore(faceinfo_file_path_hdf5) as store:
        _data_df = store['tracked_faces_dataset_01']
    
    assert participant_id in _data_df.index, f"tracking information for participant {participant_id} is not available"

    _data_df = _data_df.loc[participant_id]
    # if participant_id in _data_df.index:
    #     _except_frames.extend(list(_data_df.loc[participant_id].index))
    

    # load images
    _images = sorted(glob.glob(f"{frames_root}/*.png")) 
    if end == -1: end = len(_images)-1


    colors = {}
    thick = 2

    for i in tqdm.tqdm(range(start, end+1), total=(end-start)):
        _img_path= _images[i]
        _image_name = os.path.basename(_img_path)
        _frame, _ext = _image_name.split(".")

        if _frame in _data_df.index:
            _img = cv2.imread(_img_path)
            imgHeight, imgWidth, _ = _img.shape
            for _face_id in _data_df.loc[_frame].index:
                if not _face_id in colors:
                    color = list(np.random.random(size=3) * 256)
                    colors[_face_id] = color
                left, top, right, bottom = _data_df.loc[_frame].loc[_face_id][['left', 'top', 'right', 'bottom']].astype(np.int32)
                cv2.rectangle(_img,(left, top), (right, bottom), colors[_face_id], thickness=thick)
                cv2.putText(_img, _face_id, (left, top - 12), 0, 1e-3 * imgHeight, colors[_face_id], thick//3)
                # 
            imageOutputPath = os.path.join(tracked_faces_seq,_image_name)
            cv2.imwrite(imageOutputPath, _img)