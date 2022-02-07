#!/usr/bin/env python3
# coding: utf-8

import os
import random
import argparse
import glob
import logging
from tqdm import tqdm
import pandas as pd
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils import data
from sklearn.metrics import confusion_matrix, classification_report

from blinkdetect.models.blinkdetection import BlinkDetector
from blinkdetect.dataset import BlinkDataset1C, BlinkDataset2C, BlinkDataset4C

torch.multiprocessing.set_sharing_strategy('file_system')

def parser():
    _parser = argparse.ArgumentParser()
    _parser.add_argument("--annotation_file", required=True, type=str)
    _parser.add_argument("--model", required=True, type=str)
    _parser.add_argument("--dataset_path", required=True)
    _parser.add_argument('--dataset', required=True, choices=["BlinkingValidationSetVideos", "eyeblink8", "talkingFace", "zju", "RN"])
    _parser.add_argument("--generate_fnfp_plots", action="store_true")

    return _parser.parse_args()


# def draw_angles_dist(
#     test_yaw_pitch_rnb_fp,
#     test_yaw_pitch_rnb_fn,
#     test_yaw_pitch_rnb_tp,
#     test_yaw_pitch_rnb_tn,
#     angles_dist_file):
#     """"""
#     test_yaw_pitch_rnb_tp_tn = test_yaw_pitch_rnb_tp + test_yaw_pitch_rnb_tn
#     test_yaw_pitch_rnb_fp_fn = test_yaw_pitch_rnb_fp + test_yaw_pitch_rnb_fn
#     if len(test_yaw_pitch_rnb_fp_fn + test_yaw_pitch_rnb_tp_tn) == 0:
#         return

#     plt.axvline(0,c="k")
#     plt.axhline(0,c="k")

#     if len(test_yaw_pitch_rnb_tp) > 0:
#         _pids,_,__,_rngs, _yaw_means, _yaw_stds, _pitch_means, _pitch_stds, _scores_0, _scores_1 = test_yaw_pitch_rnb_tp
#         plt.scatter(_yaw_means, _pitch_means, s=8, marker="o", c='g', label=f"TP: {len(_pitch_stds)}")
#     if len(test_yaw_pitch_rnb_tn) > 0:
#         _pids,_,__,_rngs, _yaw_means, _yaw_stds, _pitch_means, _pitch_stds, _scores_0, _scores_1 = test_yaw_pitch_rnb_tn
#         plt.scatter(_yaw_means, _pitch_means, s=8, marker="o", c='g', label=f"TN: {len(_pitch_stds)}")
#     if len(test_yaw_pitch_rnb_fp) > 0:
#         _pids,_,__,_rngs, _yaw_means, _yaw_stds, _pitch_means, _pitch_stds, _scores_0, _scores_1 = test_yaw_pitch_rnb_fp
#         plt.scatter(_yaw_means, _pitch_means, s=6, marker="^", c="r",  label=f"FP: {len(_pitch_stds)}")
#     if len(test_yaw_pitch_rnb_fn) > 0:
#         _pids,_,__,_rngs, _yaw_means, _yaw_stds, _pitch_means, _pitch_stds, _scores_0, _scores_1 = test_yaw_pitch_rnb_fn
#         plt.scatter(_yaw_means, _pitch_means, s=6, marker="^", c="b",  label=f"FN: {len(_pitch_stds)}")

#     plt.xlabel("yaw")
#     plt.ylabel("pitch")
#     plt.legend()
#     plt.savefig(angles_dist_file, dpi=300, bbox_inches='tight')
#     plt.close()


def draw_angles_dist(y_list, y_pred_list, _yaws, _pitchs, angles_dist_file):
    status = []
    yaw = []
    pitch = []
    for x, x_pred, _yaw, _pitch in zip(y_list, y_pred_list, _yaws, _pitchs):
        if x == x_pred and x == 0: st = "TN"
        if x == x_pred and x == 1: st = "TP"
        if x != x_pred and x == 0: st = "FP"
        if x != x_pred and x == 1: st = "FN"
        status.append(st)
        yaw.append((_yaw[14] + _yaw[15]) / 2)
        pitch.append((_pitch[14] + _pitch[15]) / 2)
    df = pd.DataFrame.from_dict({"status": status, "yaw": yaw, "pitch": pitch})
    g = sns.FacetGrid(df, col="status", col_wrap=2, hue="status", col_order=["TN", "FP", "FN", "TP"])
    g.map(sns.scatterplot, "yaw", "pitch",)

    plt.savefig(angles_dist_file)


def draw_scores_dist(
    test_yaw_pitch_rnb_fp_fn,
    test_yaw_pitch_rnb_tp,
    test_yaw_pitch_rnb_tn,
    angles_dist_file):    
    if len(test_yaw_pitch_rnb_tp) > 0:
        *_, _pitch_stds, _scores_0, _scores_1 = test_yaw_pitch_rnb_tp
        plt.scatter(_scores_0, _scores_1, s=8, marker="o", c='b', label=f"TP: {len(_pitch_stds)}")
    if len(test_yaw_pitch_rnb_tn) > 0:
        *_, _pitch_stds, _scores_0, _scores_1 = test_yaw_pitch_rnb_tn
        plt.scatter(_scores_0, _scores_1, s=8, marker="o", c='r', label=f"TN: {len(_pitch_stds)}")
    if len(test_yaw_pitch_rnb_fp_fn) > 0:
        *_, _pitch_stds, _scores_0, _scores_1 = test_yaw_pitch_rnb_fp_fn
        plt.scatter(_scores_0, _scores_1, s=8, marker="^", c="k",  label=f"FP+FN: {len(_pitch_stds)}")

    if len(test_yaw_pitch_rnb_fp_fn + test_yaw_pitch_rnb_tp + test_yaw_pitch_rnb_tn) == 0:
        return

    plt.axvline(0,c="k")
    plt.axhline(0,c="k")
    plt.xlabel("scores 0")
    plt.ylabel("scores 1")
    plt.legend()
    plt.savefig(angles_dist_file, dpi=300, bbox_inches='tight')
    plt.close()


def make_gif(frame_folder, name):
    frames = [Image.open(image) for image in sorted(glob.glob(f"{frame_folder}/*.png"))]
    frame_one = frames[0]
    frame_one.save(os.path.join(frame_folder,f"{name}.gif"), format="GIF", append_images=frames,
               save_all=True, duration=50, loop=0)
    for frame_path in glob.glob(f"{frame_folder}/*.png"):
          os.remove(frame_path)


datasets_dir = os.path.join(os.path.dirname(__file__), ".." , "dataset")
BATCH_SIZE = 8

if __name__=="__main__":
    args = parser()
    _softmax = False
    # best-{args.prefix}-{args.normalized}-{args.channels}-{BATCH_SIZE}
    if len(os.path.basename(args.model).split("_")) == 3:
        fileprefix, name, step_number =os.path.basename(args.model).split("_")
        _, prefix, _normalized, _chan, _batch = fileprefix.split("-")
        if len(name.split("-")) == 2:
            _softmax = True
        _batch = "".join(["B",_batch])
    else:
        prefix, _normalized, _chan, _batch_t = os.path.basename(args.model).split("-")
        _batch, ext = _batch_t.split(".") # epoch
        _batch = "".join(["E", _batch])
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

    # logging
    log_file = os.path.join(dataset_path, f"{prefix}-{_normalized}-{_chan}-{_batch}-softmax.txt")
    if os.path.exists(log_file):
          os.remove(log_file)
    logging.basicConfig(filename=log_file,  level=logging.INFO)

    torch.manual_seed(192020)
    random.seed(192020)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    dataset_loader = torch.utils.data.DataLoader(dataset,
                                                      batch_size=BATCH_SIZE,
                                                      shuffle=True,
                                                      num_workers=1)

    # load model
    network = BlinkDetector(num_chan)
    network.load_state_dict(torch.load(args.model), strict=False)
    network.to(device)

    network.eval()

    # losses
    cls_loss = torch.nn.CrossEntropyLoss()    
    reg_loss = torch.nn.MSELoss()

    # testing
    y_pred_list = []
    y_test = []
    scores = []
    duration_MSE = 0
    classification_cls = 0
    _pids = []
    _rngs = []
    _yaws = []
    _pitchs = []

    testing_progress = tqdm(total=len(dataset_loader),
                               desc="Evaluation progress")

    with torch.no_grad():
      test_loss = 0.0
      avg_test_loss = 0.0
      for data, target, duration, _pid, _rng, _yaw, _pitch in dataset_loader:
        data, target, duration = data.to(device), target.to(device), duration.to(device)
        pred_target, pred_duration = network(data)

        scores.extend(pred_target.softmax(dim=1).cpu().numpy())

        y_test_pred = torch.max(pred_target.data,1)
        y_pred_tag = y_test_pred[1]

        y_pred_list.extend(y_pred_tag.cpu().numpy())
        y_test.extend(target.cpu().numpy())
        _pids.extend(_pid)
        _rngs.extend(_rng)
        _yaws.extend(_yaw.numpy())
        _pitchs.extend(_pitch.numpy())

        duration_MSE += reg_loss(pred_duration, duration.type_as(pred_duration)).mul(3.1355).exp_()
        classification_cls += cls_loss(pred_target, target.long())
        test_loss += cls_loss(pred_target, target.long()) + reg_loss(pred_duration, duration.type_as(pred_duration))

        testing_progress.update()

    avg_test_loss = test_loss.item()*BATCH_SIZE / len(dataset)
    avg_duration_MSE = duration_MSE.item()*BATCH_SIZE / len(dataset)
    avg_classification_cls = classification_cls.item()*BATCH_SIZE / len(dataset)
    testing_progress.set_postfix(testing_loss=avg_test_loss)

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    y_list = [a.squeeze().tolist() for a in y_test]
    scores = [a.squeeze().tolist() for a in scores]
    scores_0 = [_0 for _0, _ in scores]
    scores_1 = [_1 for _, _1 in scores]

    tn, fp, fn, tp = confusion_matrix(y_list, y_pred_list).ravel()

    # Draw_angles_dist
    angles_dist_file = os.path.join(dataset_path, f"[fp+fn]dist-{normalized}-{num_chan}-{_batch}-{name}.png")
    draw_angles_dist(y_list, y_pred_list, _yaws, _pitchs, angles_dist_file)


    # test_combined = [(x,x_pred, _id, _rng, (_yaw[14]+_yaw[15])/2, (_pitch[14]+_pitch[15])/2) for x, x_pred, _id, _rng,_yaw, _pitch in list(zip(y_list, y_pred_list, _pids, _rngs, _yaws, _pitchs)) if x!=x_pred]
    # #
    test_yaw_pitch_rnb_fp_fn = [(_pid, x,x_pred, _rng, (_yaw[14]+_yaw[15])/2, np.std(_yaw), (_pitch[14]+_pitch[15])/2, np.std(_pitch), score_0, score_1) for _pid, x, x_pred, _rng, _yaw,_pitch, score_0, score_1 in list(zip(_pids,y_list, y_pred_list, _rngs, _yaws, _pitchs, scores_0, scores_1)) if x!=x_pred]
    test_yaw_pitch_rnb_fp_fn = list(zip(*test_yaw_pitch_rnb_fp_fn))
    # # tp+tn
    # test_yaw_pitch_rnb_tp_tn = [(_pid, x,x_pred, _rng, (_yaw[14]+_yaw[15])/2, np.std(_yaw), (_pitch[14]+_pitch[15])/2, np.std(_pitch), score_0, score_1) for _pid, x, x_pred, _rng, _yaw,_pitch, score_0, score_1 in list(zip(_pids, y_list, y_pred_list, _rngs, _yaws, _pitchs, scores_0, scores_1)) if x==x_pred]
    # test_yaw_pitch_rnb_tp_tn = list(zip(*test_yaw_pitch_rnb_tp_tn))
    # New: TP
    test_yaw_pitch_rnb_tp = [(_pid, x,x_pred, _rng, (_yaw[14]+_yaw[15])/2, np.std(_yaw), (_pitch[14]+_pitch[15])/2, np.std(_pitch), score_0, score_1) for _pid, x, x_pred, _rng, _yaw,_pitch, score_0, score_1 in list(zip(_pids, y_list, y_pred_list, _rngs, _yaws, _pitchs, scores_0, scores_1)) if x==x_pred and x==1]
    test_yaw_pitch_rnb_tp = list(zip(*test_yaw_pitch_rnb_tp))
    # New: TN
    test_yaw_pitch_rnb_tn = [(_pid, x,x_pred, _rng, (_yaw[14]+_yaw[15])/2, np.std(_yaw), (_pitch[14]+_pitch[15])/2, np.std(_pitch), score_0, score_1) for _pid, x, x_pred, _rng, _yaw,_pitch, score_0, score_1 in list(zip(_pids, y_list, y_pred_list, _rngs, _yaws, _pitchs, scores_0, scores_1)) if x==x_pred and x==0]
    test_yaw_pitch_rnb_tn = list(zip(*test_yaw_pitch_rnb_tn))
    # # New: FP
    # test_yaw_pitch_rnb_fp = [(_pid, x,x_pred, _rng, (_yaw[14]+_yaw[15])/2, np.std(_yaw), (_pitch[14]+_pitch[15])/2, np.std(_pitch), score_0, score_1) for _pid, x, x_pred, _rng, _yaw,_pitch, score_0, score_1 in list(zip(_pids,y_list, y_pred_list, _rngs, _yaws, _pitchs, scores_0, scores_1)) if x!=x_pred and x_pred==1]
    # test_yaw_pitch_rnb_fp = list(zip(*test_yaw_pitch_rnb_fp))
    # # New: FN
    # test_yaw_pitch_rnb_fn = [(_pid, x,x_pred, _rng, (_yaw[14]+_yaw[15])/2, np.std(_yaw), (_pitch[14]+_pitch[15])/2, np.std(_pitch), score_0, score_1) for _pid, x, x_pred, _rng, _yaw,_pitch, score_0, score_1 in list(zip(_pids,y_list, y_pred_list, _rngs, _yaws, _pitchs, scores_0, scores_1)) if x!=x_pred and x_pred==0]
    # test_yaw_pitch_rnb_fn = list(zip(*test_yaw_pitch_rnb_fn))

    print("testing performace")
    logging.info("testing performace")
    print(f"duration mse: {avg_duration_MSE}")
    logging.info(f"duration mse: {avg_duration_MSE}")
    print(f"classifier loss: {avg_classification_cls}")
    logging.info(f"classifier loss: {avg_classification_cls}")
    print(f"tn: {tn}, fp: {fp}, fn: {fn}, tp: {tp}")
    logging.info(f"tn: {tn}, fp: {fp}, fn: {fn}, tp: {tp}")
    print(classification_report(y_list, y_pred_list))
    logging.info(classification_report(y_list, y_pred_list))

    # angles_dist_file = os.path.join(dataset_path, f"[fp+fn]dist-{normalized}-{num_chan}-{_batch}-{name}.png")
    # draw_angles_dist(test_yaw_pitch_rnb_fp, test_yaw_pitch_rnb_fn, test_yaw_pitch_rnb_tp, test_yaw_pitch_rnb_tn, angles_dist_file)
    scores_dist_file = os.path.join(dataset_path, f"[scores]dist-{normalized}-{num_chan}-{_batch}-{name}.png")
    draw_scores_dist(test_yaw_pitch_rnb_fp_fn, test_yaw_pitch_rnb_tp, test_yaw_pitch_rnb_tn, scores_dist_file)
    #

    # if args.generate_fnfp_plots:
    #     fp_fn = os.path.join(dataset_path, f"fp+fn-{normalized}-{num_chan}-{_batch}-softmax")
    #     os.makedirs(fp_fn, exist_ok=True)
    #     # TEST
    #     test_fp_fn = os.path.join(fp_fn, "test")
    #     dataset_path = os.path.dirname(dataset_path)
    #     dataset_path = os.path.join(dataset_path, "plots")
    #     sample_number = 1
    #     if len(test_combined) > 20:
    #         test_combined = random.sample(test_combined, 20)
    #     for _actual, _prediction, _pid_path, _rng, _yaw, _pitch in tqdm(test_combined, total=len(test_combined), desc="test"):
    #         # 
    #         _pid = _pid_path.replace("/","-")
    #         _start, _stop = _rng.split("-")
    #         _to = f"{test_fp_fn}/{_actual}/{_pid}_sample_{sample_number}_y{_yaw:.0f}_p{_pitch:.0f}"
    #         if not os.path.exists(_to):
    #             os.makedirs(_to)

    #         if int(_start)-5 < 1:
    #             _actual_start = 1
    #         else:
    #             _actual_start = int(_start)-5
    #         for i in range(_actual_start, int(_stop)+5):
    #             if i < 0: continue
    #             _from = os.path.join(datasets_dir, args.dataset, _pid_path, "frames", f"{i:06}.png")
    #             # _from = f"{args.dataset}/{_pid_path}/frames/{i:06}.png"
    #             _to1 = os.path.join(_to, f"{i:06}.png")
    #             # _to1 = f"{_to}/{i:06}.png"
    #             try:
    #                 shutil.copyfile(_from, _to1)
    #             except FileNotFoundError as e:
    #                 if os.path.exists(_to1):
    #                     os.remove(_to1)
    #                 continue
    #         # shutil.copyfile(_from, _to)
    #         # print(_from)
    #         make_gif(_to, f"{_pid}_[{_rng}]_{_actual}")
    #         shutil.copyfile(f"{dataset_path}/{_pid}_[{_rng}]_{_actual}.png", f"{_to}/{_pid}_[{_rng}]_{_actual}.png")
    #         sample_number+=1
