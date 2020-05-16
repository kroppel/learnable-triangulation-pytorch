#!/bin/python3

import os, sys
import cv2
import numpy as np
import pickle
from mvn.utils import vis, cfg
from mvn.datasets.cmupanoptic import CMUPanopticDataset
from mvn.utils.multiview import project_3d_points_to_image_plane_without_distortion as project

USAGE_TEXT = 'USAGE: python3 visualise_results.py <results_pkl_file> <config_yaml_file_used_in_experiment> [n_images_step=1]' 

try:
    results_file = sys.argv[1]
except:
    print("Need to specify results pkl file!")
    print(USAGE_TEXT)
    exit()

try:
    config_file = sys.argv[2]
except:
    print("Need to specify config yaml file!")
    print(USAGE_TEXT)
    exit()

try:
    n_images_step = sys.argv[3]
    if n_images_step < 1:
        raise Exception("n_images_step cannot be < 1")
except:
    n_images_step = 1

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
    image_shape=config.image_shape if hasattr(config, "image_shape") else (256, 256),
    labels_path=config.dataset.val.labels_path,
    retain_every_n_frames_in_test=config.dataset.val.retain_every_n_frames_in_test,
    scale_bbox=config.dataset.val.scale_bbox,
    square_bbox=config.dataset.val.square_bbox if hasattr(config.dataset.val, "square_bbox") else True,
    kind=config.kind,
    ignore_cameras=config.dataset.val.ignore_cameras if hasattr(config.dataset.val, "ignore_cameras") else [],
    crop=config.dataset.val.crop if hasattr(config.dataset.val, "crop") else True,
    norm_image=False
)

# Load results pkl file
with open(results_file, "rb") as f:
    data = pickle.load(f)
    keypoints3d_pred = data["keypoints_3d"]
    indexes = data["indexes"]
    images = data["images"]

camera_indexes_to_show = [0, 2, 8]

for i in range(0, len(indexes), n_images_step):
    labels = dataset[i]
    displays = []

    # Project and draw keypoints on images
    for camera_idx, camera in enumerate(labels['cameras']):
        # camera = labels['cameras'][camera_idx]
        keypoints_2d_pred = project(camera.projection, keypoints3d_pred[i][:, :3])
        keypoints_2d_gt = project(camera.projection, labels['keypoints_3d'][:, :3])

        img = labels['images'][camera_idx]

        display = vis.draw_2d_pose_cv2(keypoints_2d_pred, img, kind='coco')
        cv2.putText(display, f"Cam {camera_idx}", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))

        displays.append(display)

    # Fancy stacked images
    for i, display in enumerate(displays):
        if i == 0:
            combined = display
        else:
            combined = np.concatenate((combined, display), axis=1)

    title = f"Index {i}"
    cv2.imshow('w', combined)
    cv2.setWindowTitle('w', title)
    c = cv2.waitKey(0) % 256

    if c == ord('q') or c == 27:
        print('Quitting...')
        cv2.destroyAllWindows()
        break
print('Done')