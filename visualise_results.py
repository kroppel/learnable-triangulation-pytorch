#!/bin/python3

import os, sys
import cv2
import numpy as np
import pickle
from mvn.utils import vis, cfg
from mvn.datasets.cmupanoptic import CMUPanopticDataset

USAGE_TEXT = 'USAGE: python3 visualise_results.py <results_pkl_file> <config_file_used_in_experiment> [n_images_jump=1]' 

try:
    results_file = sys.argv[1]
except:
    print("Need to specify results pkl file!")
    print(USAGE_TEXT)
    exit()

try:
    config_file = sys.argv[2]
except:
    print("Need to specify labels npy file!")
    print(USAGE_TEXT)
    exit()

try:
    n_images_jump = sys.argv[3]
    if n_images_jump < 1:
        raise Exception("n_images_jump cannot be < 1")
except:
    n_images_jump = 1

assert os.path.exists(results_file) and os.path.isfile(results_file), f"Results file {results_file} does not exist!" 
assert os.path.exists(config_file) and os.path.isfile(config_file), f"Config file {config_file} does not exist!"

# Load config file and necessary information
config = cfg.load_config(config_file)

dataset = CMUPanopticDataset(
    cmu_root=config.dataset.val.cmu_root,
    pred_results_path=config.dataset.val.pred_results_path if hasattr(
        config.dataset.val, "pred_results_path") else None,
    train=False,
    test=True,
    image_shape=config.image_shape if hasattr(
        config, "image_shape") else (256, 256),
    labels_path=config.dataset.val.labels_path,
    retain_every_n_frames_in_test=config.dataset.val.retain_every_n_frames_in_test,
    scale_bbox=config.dataset.val.scale_bbox,
    square_bbox=config.dataset.val.square_bbox if hasattr(
        config.dataset.val, "square_bbox") else True,
    kind=config.kind,
    ignore_cameras=config.dataset.val.ignore_cameras if hasattr(
        config.dataset.val, "ignore_cameras") else [],
    crop=config.dataset.val.crop if hasattr(
        config.dataset.val, "crop") else True,
)

labels = dataset.labels

# Load results pkl file
with open(results_file, "rb") as f:
    data = pickle.load(f)
    keypoints3d_pred = data["keypoints_3d"]
    indexes = data["indexes"]
    images = data["images"]

    print(indexes)

camera_indexes_to_show = [0, 2, 8]

for i in range(len(indexes), n_images_jump):
    idx = indexes[i]
    img = images[i]
    
    shot = labels['table'][idx]
    action_idx = shot['action_idx']

    for camera_idx in camera_indexes_to_show:
        shot_camera = labels['cameras'][action_idx, camera_idx]
        camera = Camera(shot_camera['R'], shot_camera['t'], shot_camera['K'], shot_camera['dist'], camera_name)
        keypoints_2d_pred = project(camera.projection, keypoints3d_pred[i][:, :3])
        keypoints_2d_gt = project(camera.projection, labels['keypoints'][i][:, :3])

        vis.draw_2d_pose_cv2(keypoints_2d_pred, img, kind='cmu')
