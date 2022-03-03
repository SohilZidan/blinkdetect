#!/usr/bin/env python3

from typing import Tuple, List
import numpy as np
import pandas as pd


def augmnet_signal(signal: np.ndarray, label: int) -> Tuple[np.ndarray, int]:
    if label == 2:  # closing
        augmented_signal = np.flip(signal)
        augmented_label = 3
    elif label == 3:  # opening
        augmented_signal = np.flip(signal)
        augmented_label = 2
    elif label == 1:
        augmented_signal = np.flip(signal)
        augmented_label = 1
    else:
        raise(Exception(f"Not supported label value: {label}"))

    return augmented_signal, augmented_label


def load_dataset(
        annotations_file: pd.DataFrame,
        normalize=True,
        threshold: int = 300, augment: bool = False,
        leftout_annotations: List = [-1, 5]) -> Tuple[np.ndarray, np.ndarray]:
    # read annotations
    temporal_blinking = pd.read_hdf(annotations_file, "temporal_blinking")
    temporal_blinking = temporal_blinking.sort_index()

    X = []
    y = []
    y1 = []

    count_labels = dict.fromkeys([-1, 0, 1, 2, 3, 4, 5], 0)

    for sample_idx in temporal_blinking.index.unique():
        sample_signal = temporal_blinking.loc[sample_idx]

        label = sample_signal.index.unique().item()[2]
        # if label == 0:
        if label in leftout_annotations:
            continue
        if count_labels[label] == threshold:
            continue

        # input components
        eyelid_dist = sample_signal["right_eyelids_dist"] + \
            sample_signal["left_eyelids_dist"]
        eyelid_dist /= 2.

        diameter = sample_signal["right_diameter"] + \
            sample_signal["left_diameter"]
        diameter /= 2.

        pupil2corner = sample_signal["right_pupil2corner"] + \
            sample_signal["left_pupil2corner"]
        pupil2corner /= 2.

        scores = sample_signal["score"]

        if normalize:
            eyelid_dist /= diameter
            pupil2corner /= diameter
        # features
        features = np.hstack([eyelid_dist])  # , pupil2corner])

        # construct input
        X.append(features)
        y.append(label)
        y1.append(scores)
        count_labels[label] += 1
        if augment and label in [1, 2, 3]:
            eyelid_dist, label = augmnet_signal(eyelid_dist, label)
            pupil2corner, _ = augmnet_signal(pupil2corner, label)
            scores, _ = augmnet_signal(scores, label)
            features = np.hstack([eyelid_dist])  # , pupil2corner])
            X.append(features)
            y.append(label)
            y1.append(scores)
            count_labels[label] += 1

    return X, y, y1


def extract_svm_features(sample_signal):
    label = sample_signal.index.unique().item()[2]
    # if label == 0:
    # if label in leftout_annotations:
    #     continue
    # if count_labels[label] == threshold: continue

    # input components
    eyelid_dist = sample_signal["right_eyelids_dist"] + \
        sample_signal["left_eyelids_dist"]
    eyelid_dist /= 2.

    diameter = sample_signal["right_diameter"] + sample_signal["left_diameter"]
    diameter /= 2.

    pupil2corner = sample_signal["right_pupil2corner"] + \
        sample_signal["left_pupil2corner"]
    pupil2corner /= 2.

    eyelid_dist /= diameter
    pupil2corner /= diameter
    # features
    features = np.hstack([eyelid_dist, pupil2corner])
    return features, label
