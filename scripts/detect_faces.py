import argparse
import os 
import glob
from math import ceil
import pickle
import tqdm
from retinaface import RetinaFace

dataset_root = os.path.join(os.path.dirname(__file__), "..", "dataset")

def parser():
    _parser = argparse.ArgumentParser()
    _parser.add_argument('-rng', '--range', type=int, default=[0,-1], nargs=2)
    _parser.add_argument('--batch', type=int, default=32, help='number of frames to be saved as a batch')
    _parser.add_argument('--resume', action='store_true', help='if true existed frames of an existed participant will not be replaced')
    return _parser.parse_args()


# retina_extract_faces = RetinaFace.extract_faces
retina_detect_faces = RetinaFace.detect_faces
model = RetinaFace.build_model()


def extract_faces(images_paths: list, start: int = 0, end: int = -1, frames_exception: list=[], output_dir: str = './output_faces'):
    """
    """

    os.makedirs(output_dir, exist_ok=True)
    # 
    _frames_paths = []
    _frames_names = []
    faces_not_found = []
    _detections = {}
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
        # _frames_paths.append(_img_path)
        _frames_names.append(_name)

        # face detection
        dets = retina_detect_faces(img_path = _img_path, model=model)
        

        if type(dets) is tuple:
            faces_not_found=1
            n_faces = 0
        else:
            faces_not_found=0
            n_faces = len(dets.keys())
        
        _detections[_name]={
            "faces": dets, 
            "faces_not_found": faces_not_found, 
            "faces_number": n_faces
        }
        
    return _detections



if __name__=="__main__":

    args = parser()
    start, end = args.range
    resume = args.resume

    #
    all_files = glob.glob(f'{os.path.join(dataset_root, "BlinkingValidationSetVideos")}/*')
    videos_folders = [_item for _item in all_files if os.path.isdir(_item)]

    # 
    for video_folder in videos_folders:
        # input
        video_name = os.path.basename(video_folder)
        frames_root=os.path.join(video_folder, "frames")
        # output
        faces_root = os.path.join(dataset_root,"faces", video_name)
        faceinfo_file_path_pkl = os.path.join(faces_root, "faceinfo.pkl")

        os.makedirs(faces_root,exist_ok=True)

        # when resume is set, existed participant_id,frame_num indices will not be processed
        _except_frames = []
        _all_detections = {}
        if resume:
            if os.path.exists(faceinfo_file_path_pkl):
                with open( faceinfo_file_path_pkl, "rb" ) as pkl_file:
                    _all_detections = pickle.load(pkl_file)
                    _except_frames.extend(list(_all_detections.keys()))

        
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
            _new_detections = extract_faces(
                                                                images_paths=_images, 
                                                                start=_batch_start, 
                                                                end=_batch_end, 
                                                                frames_exception=_except_frames, 
                                                                output_dir=faces_root)

            _all_detections = {**_all_detections, **_new_detections}


            # save the results
            with open( faceinfo_file_path_pkl, "wb" ) as pkl_file:
                pickle.dump(_all_detections, pkl_file)

            print(f"{len(_all_detections)} detections results saved into {faceinfo_file_path_pkl}")
