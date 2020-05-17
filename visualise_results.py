#!/bin/python3

import os, sys
import cv2
import numpy as np
import pickle
import re

from mvn.utils import vis, cfg
from mvn.datasets import human36m, cmupanoptic
from mvn.utils.multiview import project_3d_points_to_image_plane_without_distortion as project

USAGE_TEXT = 'USAGE: python3 visualise_results.py <results_pkl_file> <config_yaml_file_used_in_experiment> [n_images_step=1 [save_images_instead=0]]\n NOTE: Saves images to \'saved_images\' where `results_pkl_file` is found' 

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

try:
    save_images_instead = (int(sys.argv[4]) == 1)
except:
    save_images_instead = 0

assert os.path.exists(results_file) and os.path.isfile(results_file), f"Results file {results_file} does not exist!" 
assert os.path.exists(config_file) and os.path.isfile(config_file), f"Config file {config_file} does not exist!"

# Load config file and necessary information
config = cfg.load_config(config_file)

if config.kind == "cmu":
    dataset = cmupanoptic.CMUPanopticDataset(
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
elif config.kind == "human36m" or config.kind == "h36m":
    dataset = human36m.Human36MMultiViewDataset(
        h36m_root=config.dataset.val.h36m_root,
        pred_results_path=config.dataset.val.pred_results_path if hasattr(config.dataset.val, "pred_results_path") else None,
        train=False,
        test=True,
        image_shape=config.image_shape if hasattr(config, "image_shape") else (256, 256),
        labels_path=config.dataset.val.labels_path,
        with_damaged_actions=config.dataset.val.with_damaged_actions,
        retain_every_n_frames_in_test=config.dataset.val.retain_every_n_frames_in_test,
        scale_bbox=config.dataset.val.scale_bbox,
        kind=config.kind,
        undistort_images=config.dataset.val.undistort_images,
        ignore_cameras=config.dataset.val.ignore_cameras if hasattr(config.dataset.val, "ignore_cameras") else [],
        crop=config.dataset.val.crop if hasattr(config.dataset.val, "crop") else True,
    )
else:
    raise NotImplementedError(f"{config.kind} dataset not implemented")

# Load results pkl file
with open(results_file, "rb") as f:
    data = pickle.load(f)
    keypoints3d_pred = data["keypoints_3d"]
    indexes = data["indexes"]
    images = data["images"]

img_dir = re.sub(f"{os.sep}(.+)\.pkl", "", os.path.abspath(results_file))
img_dir = os.path.join(img_dir, "saved_images")

# camera_indexes_to_show = [0, 2, 8]

for i in range(0, len(indexes), n_images_step):
    labels = dataset[i]
    displays = []

    # Project and draw keypoints on images
    for camera_idx, camera in enumerate(labels['cameras']):
        keypoints_3d_pred = keypoints3d_pred[i][:, :3]
        keypoints_3d_gt = labels['keypoints_3d'][:, :3]

        keypoints_2d_pred = project(camera.projection, keypoints_3d_pred)
        keypoints_2d_gt = project(camera.projection, keypoints_3d_gt)

        # import ipdb; ipdb.set_trace()

        img = labels['images'][camera_idx]

        display = vis.draw_2d_pose_cv2(keypoints_2d_pred, img, kind='coco')
        # display = vis.draw_2d_pose_cv2(keypoints_2d_gt, display, kind='cmu')
        cv2.putText(display, f"Cam {camera_idx:02}", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))

        displays.append(display)

    # Fancy stacked images
    for i, display in enumerate(displays):
        if i == 0:
            combined = display
        else:
            combined = np.concatenate((combined, display), axis=1)

    # Load
    if save_images_instead:
        img_path = img_dir
        if not os.path.exists(img_path):
            os.makedirs(img_path)

        img_path = os.path.join(img_path, f"{i:04}.jpg")

        try:
            print(f"Saving image to {img_path}")
            cv2.imwrite(img_path, display)
        except:
            print(f"Error: Cannot save to {img_path}")
    else:
        cv2.imshow('w', combined)
        cv2.setWindowTitle('w', f"Index {i}")
        c = cv2.waitKey(0) % 256

        if c == ord('q') or c == 27:
            print('Quitting...')
            cv2.destroyAllWindows()
            break

print('Done.')
