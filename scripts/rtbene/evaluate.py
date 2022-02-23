#!/usr/bin/env python3
# coding: utf-8

import os
import shutil
import argparse
import pickle
import numpy as np
import pandas as pd
import cv2
import tqdm
import matplotlib.pyplot as plt
from scipy.stats import kde
from dataloader import extract_svm_features
from blinkdetect.image.misc import cut_region
from sklearn.metrics import confusion_matrix, classification_report


# pose_estimator = HeadPoseEstimator()

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations",
        required=True,
        help="annotations file"
        )
    parser.add_argument(
        "--model",
        required=True,
        help="model path"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="path to the generated plots"
    )
    parser.add_argument(
        "--false_clips",
        action="store_true"
    )
    return parser.parse_args()


def draw_angles_dist(correct, correct_std, wrong, wrong_std, angles_dist_file):
    tp_tn = correct
    fp_fn = wrong
    tp_tn_std = correct_std
    fp_fn_std = wrong_std
    if len(fp_fn + tp_tn) == 0:
        return
    nbins = 20
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Yaw")
    plt.ylabel("Pitch")

    #plt.axvline(0,c="k")
    #plt.axhline(0,c="k")
    # find min and max
    xmins = []
    xmaxes = []
    ymins = []
    ymaxes = []
    for _set in [tp_tn, fp_fn]:
        _yaw_means, _pitch_means = _set
        xmins.append(np.min(_yaw_means))
        ymins.append(np.min(_pitch_means))
        xmaxes.append(np.max(_yaw_means))
        ymaxes.append(np.max(_pitch_means))
    min_x = np.min(xmins)
    min_y = np.min(ymins)
    max_x = np.max(xmaxes)
    max_y = np.max(ymaxes)


    if len(tp_tn) > 0:
        # mean
        _yaw_means, _pitch_means = tp_tn
        data_tp = np.vstack([_yaw_means, _pitch_means])
        x, y = data_tp
    
        k = kde.gaussian_kde(data_tp)
        #xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
        xi, yi = np.mgrid[min_x:max_x:nbins*1j, min_y:max_y:nbins*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        axs[0, 0]
        axs[0, 0].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.Blues, label=f"TP: {len(_yaw_means)}")
        axs[0, 0].set_aspect(1)
        axs[0, 0].set_title(f"correct (mean): {len(_yaw_means)}", fontsize='small')
        axs[0, 0].grid()
        #plt.scatter(_yaw_means, _pitch_means, s=8, marker="o", c='g', label=f"TP: {len(_tp)}")

        # std
        _yaw_means, _pitch_means = tp_tn_std
        data_tp = np.vstack([_yaw_means, _pitch_means])
        x, y = data_tp
    
        k = kde.gaussian_kde(data_tp)
        #xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
        xi, yi = np.mgrid[min_x:max_x:nbins*1j, min_y:max_y:nbins*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        axs[1, 0]
        axs[1, 0].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.Blues, label=f"TP: {len(_yaw_means)}")
        axs[1, 0].set_aspect(1)
        axs[1, 0].set_title(f"correct (std): {len(_yaw_means)}", fontsize='small')
        axs[1, 0].grid()

    if len(fp_fn) > 0:
        # mean
        _yaw_means, _pitch_means = fp_fn
        data_fp = np.vstack([_yaw_means, _pitch_means])
        x, y = data_fp

        k = kde.gaussian_kde(data_fp)

        xi, yi = np.mgrid[min_x:max_x:nbins*1j, min_y:max_y:nbins*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        axs[0, 1].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.Reds, label=f"FP: {len(_yaw_means)}")
        axs[0, 1].set_aspect(1)
        axs[0, 1].set_title(f"Wrong (mean): {len(_yaw_means)}", fontsize='small')
        axs[0, 1].grid()

        # std
        _yaw_means, _pitch_means = fp_fn_std
        data_fp = np.vstack([_yaw_means, _pitch_means])
        x, y = data_fp

        k = kde.gaussian_kde(data_fp)

        xi, yi = np.mgrid[min_x:max_x:nbins*1j, min_y:max_y:nbins*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        axs[1, 1].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.Reds, label=f"FP: {len(_yaw_means)}")
        axs[1, 1].set_aspect(1)
        axs[1, 1].set_title(f"Wrong (std): {len(_yaw_means)}", fontsize='small')
        axs[1, 1].grid()

    plt.savefig(angles_dist_file, dpi=300, bbox_inches='tight')
    plt.close()


def eval_dataset(
    annotations_file,
    model,
    output_folder,
    leftout_annotations=[-1, 5],
    draw_false_clips=False):
    # read annotations
    temporal_blinking = pd.read_hdf(annotations_file, "temporal_blinking")
    temporal_blinking = temporal_blinking.sort_index()

    # initialize
    gt_labels = []
    pred_labels = []
    yaws = []
    pitchs = []
    yaw_stds = []
    pitch_stds = []
    drownCount = 0

    if os.path.exists(f"{output_folder}-features"):
        shutil.rmtree(f"{output_folder}-features")
    if os.path.exists(f"{output_folder}"):
        shutil.rmtree(f"{output_folder}")

    unique_indexes = temporal_blinking.index.unique()
    for sample_idx in tqdm.tqdm(unique_indexes, total=unique_indexes.shape[0]):
        sample_signal = temporal_blinking.loc[sample_idx]

        features, label = extract_svm_features(sample_signal)
        pred_label = model.predict(features[:30].reshape(1,-1))
        pred_label = pred_label[0]

        # retrieve info for plotting
        frames_names = sample_signal["frame"]
        tmp_yaws = sample_signal["yaw"]
        tmp_pitchs = sample_signal["pitch"]
        bboxes = sample_signal["facial_area"]
        eyelid_dist = sample_signal["right_eyelids_dist"] + sample_signal["left_eyelids_dist"]
        eyelid_dist /= 2.
        diameter = sample_signal["right_diameter"] + sample_signal["left_diameter"]
        diameter /= 2.

        try:
            if label in leftout_annotations:
                continue
        except Exception as e:
            print(leftout_annotations)
            print(label)
            raise(e)

        os.makedirs(output_folder, exist_ok=True)

        if draw_false_clips and pred_label != label and drownCount < 1000:
            # clip folder
            clip_folder = "-".join([sample_idx[0], sample_idx[3]])

            output_dir = os.path.join(output_folder, f"GT_{label}", f"pred_{pred_label}", clip_folder)
            output_dir_w_features = os.path.join(f"{output_folder}-features", f"GT_{label}", f"pred_{pred_label}", clip_folder)
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(output_dir_w_features, exist_ok=True)
            for idx, file_path in enumerate(sample_signal["file_path"]):
                frame = cv2.imread(file_path, cv2.IMREAD_COLOR)
                file_name = os.path.basename(file_path)
                # file_dest = os.path.join(output_dir, file_name)
                # shutil.copyfile(file_path, file_dest)

                fig, axs = plt.subplots(nrows=2, ncols=1)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                _, frame = cut_region(frame, bboxes[idx])

                axs[0].imshow(frame)#, cmap='gray')
                axs[0].xaxis.set_visible(False)
                axs[0].yaxis.set_visible(False)

                axs[1].plot(frames_names, diameter, c="b")
                axs[1].scatter(frames_names, diameter, s=8, marker="o", c="b",  label="iris diameter")

                axs[1].plot(frames_names, eyelid_dist, c="r")
                axs[1].scatter(frames_names, eyelid_dist, s=8, marker="o", c="r",  label="eyelids dist")

                axs[1].plot(frames_names, features[:30], c="m")
                axs[1].scatter(frames_names, features[:30], s=8, marker="o", c="m",  label="eyelids-normalized")

                # axs[1].plot(frames_names, features[30:], c="k")
                # axs[1].scatter(frames_names, features[30:], s=8, marker="o", c="k",  label="pupil2corner-features")

                axs[1].axvline(frames_names[idx], c="k")

                fig.legend()

                new_frame_name, ext = os.path.splitext(file_name)
                frame_path_new = os.path.join(output_dir_w_features, new_frame_name+"_plots"+f"{ext}")
                plt.savefig(frame_path_new, dpi=300, bbox_inches='tight')
                plt.close(fig)
            drownCount = drownCount + 1

        yaw = np.mean(tmp_yaws)
        yaw_std = np.std(tmp_yaws)
        pitch = np.mean(tmp_pitchs)
        pitch_std = np.std(tmp_pitchs)

        gt_labels.append(label)
        pred_labels.append(pred_label)
        yaws.append(yaw)
        yaw_stds.append(yaw_std)
        pitchs.append(pitch)
        pitch_stds.append(pitch_std)

    print("confusion matrix:\n", confusion_matrix(gt_labels, pred_labels))
    print("classification report:\n", classification_report(gt_labels, pred_labels))

    # draw angle distribution
    angle_dist_file = os.path.join(output_folder, "angle_dist.png")

    correct_classified = [(yaws[idx], pitchs[idx]) for idx, gt_label in enumerate(gt_labels) if gt_label == pred_labels[idx]]
    correct_classified = list(zip(*correct_classified))
    correct_classified_std = [(yaw_stds[idx], pitch_stds[idx]) for idx, gt_label in enumerate(gt_labels) if gt_label == pred_labels[idx]]
    correct_classified_std = list(zip(*correct_classified_std))

    wrong_classified = [(yaws[idx], pitchs[idx]) for idx, gt_label in enumerate(gt_labels) if gt_label != pred_labels[idx]]
    wrong_classified = list(zip(*wrong_classified))
    wrong_classified_std = [(yaw_stds[idx], pitch_stds[idx]) for idx, gt_label in enumerate(gt_labels) if gt_label != pred_labels[idx]]
    wrong_classified_std = list(zip(*wrong_classified_std))

    draw_angles_dist(correct_classified, correct_classified_std, wrong_classified, wrong_classified_std, angle_dist_file)

    # draw score distribution


if __name__ == "__main__":
    args = parse()

    # Load from file
    with open(args.model, 'rb') as file:
        model = pickle.load(file)

    # eval
    eval_dataset(args.annotations, model, args.output, draw_false_clips=args.false_clips)