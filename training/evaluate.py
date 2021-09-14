#!/usr/bin/env python3
# coding: utf-8

import os
import argparse
from tqdm import tqdm

import torch
from torch.utils import data
import paramiko
from PIL import Image
import glob
import shutil
import random
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from blinkdetect.models.blinkdetection import BlinkDetector
from blinkdetect.dataset import BlinkDataset1C, BlinkDataset2C, BlinkDataset4C

from sklearn.metrics import confusion_matrix, classification_report

import matplotlib.pyplot as plt

def parser():
    _parser = argparse.ArgumentParser()
    _parser.add_argument("--annotation_file", required=True, type=str)
    _parser.add_argument("--model", required=True, type=str)
    _parser.add_argument("--dataset_path", required=True)
    _parser.add_argument('--dataset', required=True, choices=["BlinkingValidationSetVideos", "eyeblink8", "talkingFace", "zju", "RN"])
    _parser.add_argument("--generate_fnfp_plots", action="store_true")
    
    return _parser.parse_args()

def draw_angles_dist(test_yaw_pitch_rnb_fp_fn, test_yaw_pitch_rnb_tp_tn, angles_dist_file):
    _pids,_,__,_rngs, _yaw_means, _yaw_stds, _pitch_means, _pitch_stds = test_yaw_pitch_rnb_tp_tn
    plt.scatter(_yaw_means, _pitch_means, s=8, marker="o", c='b', label=f"TP+TN: {len(_pitch_stds)}")

    _pids,_,__,_rngs, _yaw_means, _yaw_stds, _pitch_means, _pitch_stds = test_yaw_pitch_rnb_fp_fn
    plt.scatter(_yaw_means, _pitch_means, s=10, marker="^", c="r",  label=f"FP+FN: {len(_pitch_stds)}")
    

    plt.axvline(0,c="k")
    plt.axhline(0,c="k")


    plt.xlabel("yaw")
    plt.ylabel("pitch")
    plt.legend()
    plt.savefig(angles_dist_file, dpi=300, bbox_inches='tight')
    plt.close()

def make_gif(frame_folder, name):
    frames = [Image.open(image) for image in sorted(glob.glob(f"{frame_folder}/*.png"))]
    # frames = sorted(frames)
    frame_one = frames[0]
    frame_one.save(os.path.join(frame_folder,f"{name}.gif"), format="GIF", append_images=frames,
               save_all=True, duration=50, loop=0)
    for frame_path in glob.glob(f"{frame_folder}/*.png"):
          os.remove(frame_path)

BATCH_SIZE = 8

if __name__=="__main__":
    args = parser()

    prefix, _normalized, _chan, _epochs = args.model.split("-")
    epochs, ext = _epochs.split(".")
    normalized = False if _normalized=='False' else True
    dataset_path = args.dataset_path
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    if _chan == "1C":
        num_chan = 1
        dataset = BlinkDataset1C(args.annotation_file, normalized) #Dataset class
    if _chan == "2C":
        num_chan = 2
        dataset = BlinkDataset2C(args.annotation_file, normalized) #Dataset class
    if _chan == "4C":
        num_chan = 4
        dataset = BlinkDataset4C(args.annotation_file, normalized) #Dataset class
    
    # 
    torch.manual_seed(192020)
    random.seed(192020)
    # 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 
    
    dataset_loader = torch.utils.data.DataLoader(dataset,
                                                      batch_size=BATCH_SIZE,
                                                      shuffle=True,
                                                      num_workers=1)

    # load model
    network = BlinkDetector(num_chan)
    network.load_state_dict(torch.load(args.model), strict=False)
    network.to(device)

    network.eval()

    # 
    cls_loss = torch.nn.BCEWithLogitsLoss()
    reg_loss = torch.nn.MSELoss()

    # testing
    y_pred_list = []
    y_test = []
    duration_MSE = 0
    classification_cls = 0
    _pids = []
    _rngs = []
    _yaws = []
    _pitchs = []

    # 
    testing_progress = tqdm(total=len(dataset_loader),
                               desc="Evaluation progress")
    # testing_progress.reset()
    with torch.no_grad():
      
      test_loss = 0.0
      avg_test_loss = 0.0
      for data, target, duration, _pid, _rng, _yaw, _pitch in dataset_loader:
        data, target, duration = data.to(device), target.to(device), duration.to(device)
        pred_target, pred_duration = network(data)
        # 
        y_test_pred = torch.sigmoid(pred_target)
        y_pred_tag = torch.round(y_test_pred).type_as(target)
        y_pred_list.extend(y_pred_tag.cpu().numpy())
        y_test.extend(target.cpu().numpy())
        # # # # # # # # # # #
        _pids.extend(_pid)
        _rngs.extend(_rng)
        _yaws.extend(_yaw.numpy())
        _pitchs.extend(_pitch.numpy())
        # 
        duration_MSE += reg_loss(pred_duration, duration.type_as(pred_duration)).mul(3.1355).exp_()
        classification_cls += cls_loss(pred_target, target.type_as(pred_target))
        test_loss += cls_loss(pred_target, target.type_as(pred_target)) + reg_loss(pred_duration, duration.type_as(pred_duration))
        # 
        testing_progress.update()

    avg_test_loss = test_loss.item()*BATCH_SIZE / len(dataset)
    avg_duration_MSE = duration_MSE.item()*BATCH_SIZE / len(dataset)
    avg_classification_cls = classification_cls.item()*BATCH_SIZE / len(dataset)
    testing_progress.set_postfix(testing_loss=avg_test_loss)

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    y_list = [a.squeeze().tolist() for a in y_test]
    # 
    tn, fp, fn, tp = confusion_matrix(y_list, y_pred_list).ravel()

    # 
    test_combined = [(x,x_pred, _id, _rng, (_yaw[14]+_yaw[15])/2, (_pitch[14]+_pitch[15])/2) for x, x_pred, _id, _rng,_yaw, _pitch in list(zip(y_list, y_pred_list, _pids, _rngs, _yaws, _pitchs)) if x!=x_pred]
    # 
    test_yaw_pitch_rnb_fp_fn = [(_pid, x,x_pred, _rng, (_yaw[14]+_yaw[15])/2, np.std(_yaw), (_pitch[14]+_pitch[15])/2, np.std(_pitch)) for _pid, x, x_pred, _rng, _yaw,_pitch in list(zip(_pids,y_list, y_pred_list, _rngs, _yaws, _pitchs)) if x!=x_pred]
    # for i,t in enumerate(_yaws):
    #       if np.mean(t) > 45:
    #         print(y_list[i], y_pred_list[i], t.shape, np.mean(t), t)
    # exit()
    test_yaw_pitch_rnb_fp_fn = list(zip(*test_yaw_pitch_rnb_fp_fn))
    # tp+tn
    test_yaw_pitch_rnb_tp_tn = [(_pid, x,x_pred, _rng, (_yaw[14]+_yaw[15])/2, np.std(_yaw), (_pitch[14]+_pitch[15])/2, np.std(_pitch)) for _pid, x, x_pred, _rng, _yaw,_pitch in list(zip(_pids, y_list, y_pred_list, _rngs, _yaws, _pitchs)) if x==x_pred]
    test_yaw_pitch_rnb_tp_tn = list(zip(*test_yaw_pitch_rnb_tp_tn))

    angles_dist_file = os.path.join(dataset_path, f"[fp+fn]dist-{normalized}-{num_chan}-{epochs}.png")
    draw_angles_dist(test_yaw_pitch_rnb_fp_fn, test_yaw_pitch_rnb_tp_tn,angles_dist_file)


    print("testing performace")
    # logging.info("testing performace")
    print(f"duration mse: {avg_duration_MSE}")
    # logging.info(f"duration mse: {avg_duration_MSE}")
    print(f"classifier loss: {avg_classification_cls}")
    # logging.info(f"classifier loss: {avg_classification_cls}")
    print(f"tn: {tn}, fp: {fp}, fn: {fn}, tp: {tp}")
    # logging.info(f"tn: {tn}, fp: {fp}, fn: {fn}, tp: {tp}")
    print(classification_report(y_list, y_pred_list))
    # logging.info(classification_report(y_list, y_pred_list))

    fp_fn = os.path.join(dataset_path, f"fp+fn-{normalized}-{num_chan}-{epochs}")
    os.makedirs(fp_fn, exist_ok=True)

    # # # # # # # # # # # # # # # # # # # # # # # 
    
    if args.generate_fnfp_plots:
        paramiko.util.log_to_file('logfile.log')
        host = "nipg11.inf.elte.hu"
        port = 10113
        transport = paramiko.Transport((host, port))
        password = "S9hiLLai*"
        username = "zidan"
        transport.connect(username = username, password = password)

        sftp = paramiko.SFTPClient.from_transport(transport)
        # print(type(sftp), sftp.getcwd())
        # print(sftp.get_channel())
        # print(sftp.listdir())
        # # # # # # # # # # # # # # # # # # # # # # # 

        # TEST
        test_fp_fn = os.path.join(fp_fn, "test")
        dataset_path = os.path.dirname(dataset_path)
        dataset_path = os.path.join(dataset_path, "plots")
        sample_number = 1
        if len(test_combined) > 100:
            test_combined = random.sample(test_combined, 100)
        for _actual, _prediction, _pid, _rng, _yaw, _pitch in tqdm(test_combined, total=len(test_combined), desc="test"):
            # 
            _start, _stop = _rng.split("-")
            _to = f"{test_fp_fn}/{_actual}/{_pid}_sample_{sample_number}_y{_yaw:.0f}_p{_pitch:.0f}"
            if not os.path.exists(_to):
                os.makedirs(_to)
            
            # 
            for i in range(int(_start)-5, int(_stop)+5):
                if i < 0: continue
                _from = f"./blinkdetection/dataset/{args.dataset}/{_pid}/frames/{i:06}.png"
                _to1 = f"{_to}/{i:06}.png"
                sftp.get(_from, _to1)
            # shutil.copyfile(_from, _to)
            make_gif(_to,f"{_pid}_[{_rng}]_{_actual}")
            shutil.copyfile(f"{dataset_path}/{_pid}_[{_rng}]_{_actual}.png", f"{_to}/{_pid}_[{_rng}]_{_actual}.png")
            sample_number+=1