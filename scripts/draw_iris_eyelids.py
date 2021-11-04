import argparse
import os
import shutil
import tqdm
import cv2

# TO BE EDITED
from blinkdetect.eyelandmarks import IrisHandler        
        

def _parser():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--frames", required=True, type=str)
    arg_parser.add_argument("--output", required=True, type=str)
    arg_parser.add_argument("--facemesh_path", type=str)
    arg_parser.add_argument("--iris_path", type=str)
    return arg_parser.parse_args()


if __name__=="__main__":
    args = _parser()
    new_dir = args.output
    #
    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)
    os.makedirs(new_dir)

    assert os.path.exists(args.facemesh_path),
        f"{args.facemesh_path} does not exist"
    assert os.path.exists(args.iris_path),
        f"{args.iris_path} does not exist"

    iris_marker = IrisMarker(args.facemesh_path, args.iris_path)
    frames = sorted(os.listdir(args.frames))
    for frame in tqdm.tqdm(frames, total=len(frames)):
        name, ext = os.path.splitext(frame)
        frame_path = os.path.join(args.frames,frame)

        img = cv2.imread(frame_path)

        img = iris_marker.overlay_image(img, transform=True)

        frame_path = os.path.join(new_dir,frame)
        cv2.imwrite(frame_path, img)