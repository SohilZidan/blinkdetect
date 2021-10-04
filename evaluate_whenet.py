#!/usr/bin/env python3
# coding: utf-8

"""
	This script evaluate WHENET model against CelebAMask-HQ dataset 
	using Absolute Wrapped Error as a loss function
"""
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
    # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e: # Memory growth must be set before GPUs have been initialized
        print(e)

 # Head pose (a) is parameterized by pitch (red-axis), yaw (green-axis) and roll (blueaxis) angles in indicated directions
 # MSE
 # L_wrap
# 
import csv
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import os
import cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), './HeadPoseEstimation-WHENet'))
from whenet import WHENet


dataset_path = "dataset/CelebAMask-HQ/"
images_folder = "CelebA-HQ-img/"

# def loss_wrap():
# 	pass

# def loss_mse(gt, pred):
# 	# pass
# 	# _gt, _pred = np.array(gt), np.array(pred)
# 	return np.square(np.subtract(gt,pred)).mean()

# def loss_rmse():
# 	return np.sqrt(np.square(np.subtract(gt,pred))).mean()


def loss_AWE(gts, preds):
	'''
	 Absolute Wrapped Error for either of the three axes
	'''
	# pass
	diff = np.absolute(np.subtract(gts,preds))
	period_diff = 360 - diff
	return np.minimum(diff, period_diff)

def loss_MAWE(awes):
	'''
	Mean AWE
	'''
	return awes.mean(axis=0)


def read_gt():
	df = pd.read_csv(dataset_path+"CelebAMask-HQ-pose-anno.txt", delimiter=' ', skiprows=0, header=1, dtype={'file': str, 'Yaw': np.float64, 'Pitch': np.float64, 'Raw': np.float64})
	# modify file path
	df['file'] = df['file'].apply(lambda file_name: dataset_path+images_folder+file_name)
	df['Yaw_pred'] = 0.0
	df['Pitch_pred'] = 0.0
	df['Raw_pred'] = 0.0
	# print(df.head())
	# print(df.iloc[0][['Yaw', 'Pitch', 'Raw']].values)
	# print(df.iloc[0][['Yaw', 'Raw', 'Pitch']].values)
	return df


def pred(img_path, model):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb,(224,224))
    img_rgb = np.expand_dims(img_rgb, axis=0)
    # yaw, pitch, roll = model.get_angle(img_rgb)
    return model.get_angle(img_rgb)

# if os.path.isfile('pred-gt.csv'): exit()


if __name__ == "__main__":	
    if os.path.isfile('pred-gt.csv'):
    	data_df = pd.read_csv('pred-gt.csv')
    	print(data_df['Yaw'].shape)
    	yaw_mawe = loss_MAWE(loss_AWE(data_df['Yaw'], data_df['Yaw_pred']))
    	print("Max Yaw:", np.max(data_df['Yaw']), "Min:", np.min(data_df['Yaw']))
    	print("Max Pitch:", np.max(data_df['Pitch']), "Min:", np.min(data_df['Pitch']))
    	print("Max Roll:", np.max(data_df['Raw']), "Min:", np.min(data_df['Raw']))
    	pitch_mawe = loss_MAWE(loss_AWE(data_df['Pitch'], data_df['Pitch_pred']))
    	raw_mawe = loss_MAWE(loss_AWE(data_df['Raw'], data_df['Raw_pred']))
    	print("Yaw =", yaw_mawe)
    	print("Pitch =",pitch_mawe)
    	print("Raw =", raw_mawe)
    	print("MAE (Mean Average Error of Yaw, Pitch and Raw) =", np.array([yaw_mawe, pitch_mawe, raw_mawe]).mean())
    	exit()

    model = WHENet('./HeadPoseEstimation-WHENet/WHENet.h5')
    # print(model.model.summary())
    # 
    data_df = read_gt()

    for i, row in tqdm(data_df.iterrows(), total=len(data_df), desc='Predicting'):
    	img_path = row['file']
    	# predict
    	yaw, pitch, raw = pred(img_path, model)
    	# store
    	data_df.loc[i, 'Yaw_pred'] = yaw
    	data_df.loc[i, 'Pitch_pred'] = pitch 
    	data_df.loc[i, 'Raw_pred'] = raw

    # save to csv
    data_df.to_csv('pred-gt.csv')