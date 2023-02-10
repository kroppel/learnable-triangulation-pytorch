import os
import shutil
import argparse
import time
import json
from datetime import datetime
from collections import defaultdict
from itertools import islice
import pickle
import copy

import numpy as np
import cv2

import torch
from torch import nn
from torch import autograd
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

from tensorboardX import SummaryWriter

from mvn.models.triangulation import RANSACTriangulationNet, AlgebraicTriangulationNet, VolumetricTriangulationNet
from mvn.models.loss import KeypointsMSELoss, KeypointsMSESmoothLoss, KeypointsMAELoss, KeypointsL2Loss, VolumetricCELoss

from mvn.utils import img, multiview, op, vis, misc, cfg
from mvn.datasets import human36m, cmupanoptic, cmupanoptic_pose3
from mvn.datasets import utils as dataset_utils

# need this to overcome overflow error with pickling
import pickle4reducer 

DEBUG = False

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True, help="Path, where config file is stored")
    parser.add_argument('--eval', action='store_true', help="If set, then only evaluation will be done")
    parser.add_argument('--eval_dataset', type=str, default='val', help="Dataset split on which evaluate. Can be 'train' and 'val'")

    parser.add_argument("--local_rank", type=int, help="Local rank of the process on the node")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    parser.add_argument("--logdir", type=str, default="./logs", help="Path, where logs will be stored")

    args = parser.parse_args()
    return args


def setup_human36m_dataloaders(config, is_train, distributed_train):
    train_dataloader = None
    if is_train:
        # train
        train_dataset = human36m.Human36MMultiViewDataset(
            h36m_root=config.dataset.train.h36m_root,
            pred_results_path=config.dataset.train.pred_results_path if hasattr(config.dataset.train, "pred_results_path") else None,
            train=True,
            test=False,
            image_shape=config.image_shape if hasattr(config, "image_shape") else (256, 256),
            labels_path=config.dataset.train.labels_path,
            with_damaged_actions=config.dataset.train.with_damaged_actions,
            scale_bbox=config.dataset.train.scale_bbox,
            kind=config.kind,
            undistort_images=config.dataset.train.undistort_images,
            ignore_cameras=config.dataset.train.ignore_cameras if hasattr(config.dataset.train, "ignore_cameras") else [],
            crop=config.dataset.train.crop if hasattr(config.dataset.train, "crop") else True,
        )

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed_train else None

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.opt.batch_size,
            shuffle=config.dataset.train.shuffle and (train_sampler is None), # debatable
            sampler=train_sampler,
            collate_fn=dataset_utils.make_collate_fn(randomize_n_views=config.dataset.train.randomize_n_views,
                                                     min_n_views=config.dataset.train.min_n_views,
                                                     max_n_views=config.dataset.train.max_n_views),
            num_workers=config.dataset.train.num_workers,
            worker_init_fn=dataset_utils.worker_init_fn,
            pin_memory=True,
            drop_last=False
        )

    # val
    val_dataset = human36m.Human36MMultiViewDataset(
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

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.opt.val_batch_size if hasattr(config.opt, "val_batch_size") else config.opt.batch_size,
        shuffle=config.dataset.val.shuffle,
        collate_fn=dataset_utils.make_collate_fn(randomize_n_views=config.dataset.val.randomize_n_views,
                                                 min_n_views=config.dataset.val.min_n_views,
                                                 max_n_views=config.dataset.val.max_n_views),
        num_workers=config.dataset.val.num_workers,
        worker_init_fn=dataset_utils.worker_init_fn,
        pin_memory=True,
        drop_last=False
    )

    return train_dataloader, val_dataloader, train_sampler


def setup_cmu_dataloaders(config, is_train, distributed_train):
    train_dataloader = None
    if is_train:
        # train
        train_dataset = cmupanoptic.CMUPanopticDataset(
            cmu_root=config.dataset.train.cmu_root,
            pred_results_path=config.dataset.train.pred_results_path if hasattr(config.dataset.train, "pred_results_path") else None,
            train=True,
            test=False,
            image_shape=config.image_shape if hasattr(config, "image_shape") else (256, 256),
            labels_path=config.dataset.train.labels_path,
            scale_bbox=config.dataset.train.scale_bbox,
            square_bbox=config.dataset.train.square_bbox if hasattr(config.dataset.train, "square_bbox") else True,
            kind=config.kind,
            transfer_cmu_to_human36m=config.model.transfer_cmu_to_human36m if hasattr(config.model, "transfer_cmu_to_human36m") else False,
            choose_cameras=config.dataset.train.choose_cameras if hasattr(config.dataset.train, "choose_cameras") else [],
            ignore_cameras=config.dataset.train.ignore_cameras if hasattr(config.dataset.train, "ignore_cameras") else [],
            crop=config.dataset.train.crop if hasattr(config.dataset.train, "crop") else True,
            frames_split_file=config.opt.frames_split_file if hasattr(config.opt, "frames_split_file") else None
        )

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed_train else None

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.opt.batch_size,
            shuffle=config.dataset.train.shuffle and (train_sampler is None),  # debatable
            sampler=train_sampler,
            collate_fn=dataset_utils.make_collate_fn(randomize_n_views=config.dataset.train.randomize_n_views,
                                                     min_n_views=config.dataset.train.min_n_views,
                                                     max_n_views=config.dataset.train.max_n_views),
            num_workers=config.dataset.train.num_workers,
            worker_init_fn=dataset_utils.worker_init_fn,
            pin_memory=True,
            drop_last=False
        )

    # val
    val_dataset = cmupanoptic.CMUPanopticDataset(
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
        choose_cameras=config.dataset.val.choose_cameras if hasattr(config.dataset.val, "choose_cameras") else [],
        ignore_cameras=config.dataset.val.ignore_cameras if hasattr(config.dataset.val, "ignore_cameras") else [],
        crop=config.dataset.val.crop if hasattr(config.dataset.val, "crop") else True,
        frames_split_file=config.opt.frames_split_file if hasattr(config.opt, "frames_split_file") else None
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.opt.val_batch_size if hasattr(config.opt, "val_batch_size") else config.opt.batch_size,
        shuffle=config.dataset.val.shuffle,
        collate_fn=dataset_utils.make_collate_fn(randomize_n_views=config.dataset.val.randomize_n_views,
                                                 min_n_views=config.dataset.val.min_n_views,
                                                 max_n_views=config.dataset.val.max_n_views),
        num_workers=config.dataset.val.num_workers,
        worker_init_fn=dataset_utils.worker_init_fn,
        pin_memory=True,
        drop_last=False
    )

    return train_dataloader, val_dataloader, train_sampler

def setup_cmu_pose3_dataloaders(config, is_train, distributed_train):
    train_dataloader = None
    if is_train:
        # train
        train_dataset = cmupanoptic_pose3.CMUPanopticPose3Dataset(
            cmu_root=config.dataset.train.cmu_root,
            pred_results_path=config.dataset.train.pred_results_path if hasattr(config.dataset.train, "pred_results_path") else None,
            train=True,
            test=False,
            image_shape=config.image_shape if hasattr(config, "image_shape") else (256, 256),
            labels_path=config.dataset.train.labels_path,
            scale_bbox=config.dataset.train.scale_bbox,
            square_bbox=config.dataset.train.square_bbox if hasattr(config.dataset.train, "square_bbox") else True,
            kind=config.kind,
            transfer_cmu_to_human36m=config.model.transfer_cmu_to_human36m if hasattr(config.model, "transfer_cmu_to_human36m") else False,
            choose_cameras=config.dataset.train.choose_cameras if hasattr(config.dataset.train, "choose_cameras") else [],
            ignore_cameras=config.dataset.train.ignore_cameras if hasattr(config.dataset.train, "ignore_cameras") else [],
            crop=config.dataset.train.crop if hasattr(config.dataset.train, "crop") else True,
            frames_split_file=config.opt.frames_split_file if hasattr(config.opt, "frames_split_file") else None
        )

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed_train else None

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.opt.batch_size,
            shuffle=config.dataset.train.shuffle and (train_sampler is None),  # debatable
            sampler=train_sampler,
            collate_fn=dataset_utils.make_collate_fn(randomize_n_views=config.dataset.train.randomize_n_views,
                                                     min_n_views=config.dataset.train.min_n_views,
                                                     max_n_views=config.dataset.train.max_n_views),
            num_workers=config.dataset.train.num_workers,
            worker_init_fn=dataset_utils.worker_init_fn,
            pin_memory=True,
            drop_last=False
        )

    # val
    val_dataset = cmupanoptic_pose3.CMUPanopticPose3Dataset(
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
        choose_cameras=config.dataset.val.choose_cameras if hasattr(config.dataset.val, "choose_cameras") else [],
        ignore_cameras=config.dataset.val.ignore_cameras if hasattr(config.dataset.val, "ignore_cameras") else [],
        crop=config.dataset.val.crop if hasattr(config.dataset.val, "crop") else True,
        frames_split_file=config.opt.frames_split_file if hasattr(config.opt, "frames_split_file") else None
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.opt.val_batch_size if hasattr(config.opt, "val_batch_size") else config.opt.batch_size,
        shuffle=config.dataset.val.shuffle,
        collate_fn=dataset_utils.make_collate_fn(randomize_n_views=config.dataset.val.randomize_n_views,
                                                 min_n_views=config.dataset.val.min_n_views,
                                                 max_n_views=config.dataset.val.max_n_views),
        num_workers=config.dataset.val.num_workers,
        worker_init_fn=dataset_utils.worker_init_fn,
        pin_memory=True,
        drop_last=False
    )

    return train_dataloader, val_dataloader, train_sampler



def setup_dataloaders(config, is_train=True, distributed_train=False):
    if config.dataset.kind == 'human36m':
        train_dataloader, val_dataloader, train_sampler = setup_human36m_dataloaders(config, is_train, distributed_train)
    elif config.dataset.kind in ['cmu', 'cmupanoptic']:
        train_dataloader, val_dataloader, train_sampler = setup_cmu_dataloaders(config, is_train, distributed_train)
    elif config.dataset.kind == 'cmu_pose3':
        train_dataloader, val_dataloader, train_sampler = setup_cmu_pose3_dataloaders(config, is_train, distributed_train)
        # train_dataloader, val_dataloader, train_sampler = setup_example_dataloaders(config, is_train, distributed_train)
    else:
        raise NotImplementedError("Unknown dataset: {}".format(config.dataset.kind))

    return train_dataloader, val_dataloader, train_sampler


def setup_experiment(config, model_name, is_train=True):
    prefix = "" if is_train else "eval_"

    if config.title:
        experiment_title = config.title + "_" + model_name
    else:
        experiment_title = model_name

    experiment_title = prefix + experiment_title

    experiment_name = '{}@{}'.format(experiment_title, datetime.now().strftime("%d.%m.%Y-%H:%M:%S"))
    print("Experiment name: {}".format(experiment_name))

    experiment_dir = os.path.join(args.logdir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    shutil.copy(args.config, os.path.join(experiment_dir, "config.yaml"))

    # tensorboard
    writer = SummaryWriter(os.path.join(experiment_dir, "tb"))

    # dump config to tensorboard
    writer.add_text(misc.config_to_str(config), "config", 0)

    return experiment_dir, writer


def one_epoch(model, criterion, opt, config, dataloader, epoch, n_iters_total=0, is_train=True, caption='', master=False, experiment_dir=None, writer=None):
    name = "train" if is_train else "val"
    model_type = config.model.name

    if is_train:
        model.train()
    else:
        model.eval()

    metric_dict = defaultdict(list)

    results = defaultdict(list)
    
    save_extra_data = config.save_extra_data if hasattr(config, "save_extra_data") else False
    
    if save_extra_data:
        extra_data = defaultdict(list)

    transfer_cmu_h36m = config.model.transfer_cmu_to_human36m if hasattr(config.model, "transfer_cmu_to_human36m") else False

    print("Transfer CMU to H36M: ", transfer_cmu_h36m)
    print("Using GT Pelvis position: ", config.model.use_gt_pelvis if hasattr(config.model, "use_gt_pelvis") else False)
    print("Using cameras: ", dataloader.dataset.choose_cameras if hasattr(dataloader.dataset, "choose_cameras") else False)
    print("Debug Mode: ", DEBUG)
    print("Training: ", is_train)

    train_eval_mode = "Train" if is_train else "Eval"

    # used to turn on/off gradients
    grad_context = torch.autograd.enable_grad if is_train else torch.no_grad
    with grad_context():
        end = time.time()

        iterator = enumerate(dataloader)

        if is_train and config.opt.n_iters_per_epoch is not None:
            iterator = islice(iterator, config.opt.n_iters_per_epoch)

        if not is_train and config.opt.n_iters_per_epoch_val is not None:
            iterator = islice(iterator, config.opt.n_iters_per_epoch_val)

        '''
        Data breakdown:
        - For each of the (max) 31 cameras in CMU dataset:
            - OpenCV Image: Numpy array [Note: likely cropped to smaller shape]
            - BBOX Detection for the image: (left, top, right, bottom) tuple
            - Camera: `Camera` object from `multiview.py`
        - Index: int
        - Keypoints (gt): NP Array, (17, 4)
        - Keypoints (pred): NP Array, (17, 4) [Note: may not be there]
        '''
        ignore_batch = [ ]

        for iter_i, batch in iterator:
            print("BATCH NR {}".format(iter_i))
            if not is_train and iter_i in ignore_batch:
                continue

            if True: # with autograd.detect_anomaly():
                # measure data loading time
                data_time = time.time() - end
                    
                if batch is None:
                    print(
                        f"[{train_eval_mode}, {epoch}] Found None batch: {iter_i}")
                    continue
                
                if DEBUG:                    
                    print(f"{train_eval_mode} batch {iter_i}...")
                    print(f"[{train_eval_mode}, {epoch}, {iter_i}] Preparing batch... ", end="")

                images_batch, keypoints_3d_gt, keypoints_3d_validity_gt, proj_matricies_batch = dataset_utils.prepare_batch(batch, config)

                if DEBUG: 
                    print("Prepared!")
                

                if DEBUG: 
                    print(f"[{train_eval_mode}, {epoch}, {iter_i}] Running {model_type} model... ", end="")

                keypoints_2d_pred, cuboids_pred, base_points_pred = None, None, None
                if model_type == "alg" or model_type == "ransac":
                    keypoints_3d_pred, keypoints_2d_pred, heatmaps_pred, confidences_pred = model(images_batch, proj_matricies_batch, batch)
                elif model_type == "vol":
                    keypoints_3d_pred, heatmaps_pred, volumes_pred, confidences_pred, cuboids_pred, coord_volumes_pred, base_points_pred = model(images_batch, proj_matricies_batch, batch)
                else:
                    raise NotImplementedError(f"Unknown model type {model_type}")

                if DEBUG:
                    print("Done!")

                # batch shape[2] is likely to be the number of channels
                # n_views is also the number of cameras being used in this batch 
                batch_size, n_views, image_shape = images_batch.shape[0], images_batch.shape[1], tuple(images_batch.shape[3:])
                n_joints = keypoints_3d_pred.shape[1]

                keypoints_3d_binary_validity_gt = (keypoints_3d_validity_gt > 0.0).type(torch.float32)

                # Due to differences in model used, it may be possible that the gt and pred keypoints have different scales
                # Set this difference in scaling in the config.yaml file
                scale_keypoints_3d = config.opt.scale_keypoints_3d if hasattr(config.opt, "scale_keypoints_3d") else 1.0
                scale_keypoints_3d_gt = config.opt.scale_keypoints_3d_gt if hasattr(config.opt, "scale_keypoints_3d_gt") else scale_keypoints_3d

                # force ground truth keypoints to fit config kind
                keypoints_gt_original = keypoints_3d_gt.clone()

                if keypoints_3d_gt.shape[1] != n_joints : #and transfer_cmu_h36m:
                    print(
                        f"[Warning] Possibly due to different pretrained model type, ground truth has {keypoints_3d_gt.shape[1]} keypoints while predicted has {n_joints} keypoints"
                    )
                    keypoints_3d_gt = keypoints_3d_gt[:, :n_joints, :]
                    keypoints_3d_binary_validity_gt = keypoints_3d_binary_validity_gt[
                        :, :n_joints, :]

                # 1-view case
                # TODO: Totally remove for CMU dataset (which doesnt have pelvis-offset errors)?
                if n_views == 1:
                    print(f"[{train_eval_mode}, {epoch}, {iter_i}] {config.kind} 1-view case: batch {iter_i}, images {images_batch.shape}")

                    if config.kind == "human36m":
                        base_joint = 6
                    elif config.kind in ["coco", "cmu", "cmupanoptic", "cmu_pose3"]:
                        base_joint = 11

                    keypoints_3d_gt_transformed = keypoints_3d_gt.clone()
                    keypoints_3d_gt_transformed[:, torch.arange(n_joints) != base_joint] -= keypoints_3d_gt_transformed[:, base_joint:base_joint + 1]
                    keypoints_3d_gt = keypoints_3d_gt_transformed
                    
                    keypoints_3d_pred_transformed = keypoints_3d_pred.clone()
                    keypoints_3d_pred_transformed[:, torch.arange(n_joints) != base_joint] -= keypoints_3d_pred_transformed[:, base_joint:base_joint + 1]
                    keypoints_3d_pred = keypoints_3d_pred_transformed

                # calculate loss
                if DEBUG:
                    print(f"[{train_eval_mode}, {epoch}, {iter_i}] Calculating loss... ", end="")

                total_loss = 0.0

                loss = criterion(
                    keypoints_3d_pred * scale_keypoints_3d, 
                    keypoints_3d_gt * scale_keypoints_3d, 
                    keypoints_3d_binary_validity_gt
                )
                total_loss += loss
                metric_dict[f'{config.opt.criterion}'].append(loss.item())

                # volumetric ce loss
                use_volumetric_ce_loss = config.opt.use_volumetric_ce_loss if hasattr(config.opt, "use_volumetric_ce_loss") else False
                if use_volumetric_ce_loss:
                    volumetric_ce_criterion = VolumetricCELoss()

                    loss = volumetric_ce_criterion(coord_volumes_pred, volumes_pred, keypoints_3d_gt, keypoints_3d_binary_validity_gt)
                    metric_dict['volumetric_ce_loss'].append(loss.item())

                    weight = config.opt.volumetric_ce_loss_weight if hasattr(config.opt, "volumetric_ce_loss_weight") else 1.0
                    total_loss += weight * loss

                metric_dict['total_loss'].append(total_loss.item())

                if DEBUG:
                    print("Done!")

                if is_train:
                    if DEBUG:
                        print(f"[{train_eval_mode}, {epoch}, {iter_i}] Backpropragating... ", end="")

                    opt.zero_grad()
                    total_loss.backward()

                    if hasattr(config.opt, "grad_clip"):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.opt.grad_clip / config.opt.lr)

                    metric_dict['grad_norm_times_lr'].append(config.opt.lr * misc.calc_gradient_norm(filter(lambda x: x[1].requires_grad, model.named_parameters())))

                    opt.step()

                    if DEBUG:
                        print("Done!")

                # calculate metrics
                if DEBUG:
                    print(f"[{train_eval_mode}, {epoch}, {iter_i}] Calculating metrics... ", end="")

                l2 = KeypointsL2Loss()(
                    keypoints_3d_pred * scale_keypoints_3d,
                    keypoints_3d_gt * scale_keypoints_3d,
                    keypoints_3d_binary_validity_gt
                )
                metric_dict['l2'].append(l2.item())

                # base point l2
                if base_points_pred is not None:
                    if DEBUG:
                        print(f"\n\tCalculating base point metric...", end="")

                    base_point_l2_list = []
                    for batch_i in range(batch_size):
                        base_point_pred = base_points_pred[batch_i]

                        if config.model.kind == "coco":
                            base_point_gt = (keypoints_3d_gt[batch_i, 11, :3] + keypoints_3d_gt[batch_i, 12, :3]) / 2
                        elif config.model.kind == "mpii":
                            base_point_gt = keypoints_3d_gt[batch_i, 6, :3]
                        elif config.model.kind == "cmu":
                            base_point_gt = keypoints_3d_gt[batch_i, 2, :3]

                        base_point_l2_list.append(torch.sqrt(torch.sum((base_point_pred * scale_keypoints_3d - base_point_gt * scale_keypoints_3d) ** 2)).item())

                    base_point_l2 = 0.0 if len(base_point_l2_list) == 0 else np.mean(base_point_l2_list)
                    metric_dict['base_point_l2'].append(base_point_l2)

                    if DEBUG:
                        print("Done!")

                if DEBUG:
                    print("Done!")

                # save answers for evalulation
                if not is_train:
                    results['keypoints_3d'].append(keypoints_3d_pred.detach().cpu().numpy())
                    results['indexes'].append(batch['indexes'])
                    
                    if save_extra_data:
                        extra_data['images'].append(batch['images'])
                        extra_data['detections'].append(batch['detections'])
                        extra_data['keypoints_3d_gt'].append(batch['keypoints_3d'])
                        extra_data['cameras'].append(batch['cameras'])

                # plot visualization
                # NOTE: transfer_cmu_h36m has a visualisation error, and connectivity dict needs to be h36m
                if master:
                    if n_iters_total % config.vis_freq == 0:# or total_l2.item() > 500.0:
                        vis_kind = config.kind if hasattr(config, "kind") else "coco"
                        pred_kind = config.pred_kind if hasattr(config, "pred_kind") else None

                        if transfer_cmu_h36m and pred_kind is None:
                            pred_kind = "human36m"
                        
                        # NOTE: Because of transfering, using original gt instead of truncated ones 
                        for batch_i in range(min(batch_size, config.vis_n_elements)):
                            keypoints_vis = vis.visualize_batch(
                                images_batch, heatmaps_pred, keypoints_2d_pred, proj_matricies_batch,
                                keypoints_gt_original, keypoints_3d_pred,
                                kind=vis_kind,
                                cuboids_batch=cuboids_pred,
                                confidences_batch=confidences_pred,
                                batch_index=batch_i, size=5,
                                max_n_cols=10,
                                pred_kind=pred_kind
                            )
                            writer.add_image(f"{name}/keypoints_vis/{batch_i}", keypoints_vis.transpose(2, 0, 1), global_step=n_iters_total)

                            heatmaps_vis = vis.visualize_heatmaps(
                                images_batch, heatmaps_pred,
                                kind=pred_kind,
                                batch_index=batch_i, size=5,
                                max_n_rows=10, max_n_cols=10
                            )
                            writer.add_image(f"{name}/heatmaps/{batch_i}", heatmaps_vis.transpose(2, 0, 1), global_step=n_iters_total)

                            if model_type == "vol":
                                volumes_vis = vis.visualize_volumes(
                                    images_batch, volumes_pred, proj_matricies_batch,
                                    kind=pred_kind,
                                    cuboids_batch=cuboids_pred,
                                    batch_index=batch_i, size=5,
                                    max_n_rows=1, max_n_cols=16
                                )
                                writer.add_image(f"{name}/volumes/{batch_i}", volumes_vis.transpose(2, 0, 1), global_step=n_iters_total)

                    # dump weights to tensoboard
                    if n_iters_total % config.vis_freq == 0:
                        for p_name, p in model.named_parameters():
                            try:
                                writer.add_histogram(p_name, p.clone().cpu().data.numpy(), n_iters_total)
                            except ValueError as e:
                                print(e)
                                print(p_name, p)
                                exit()

                    # dump to tensorboard per-iter loss/metric stats
                    if is_train:
                        for title, value in metric_dict.items():
                            writer.add_scalar(f"{name}/{title}", value[-1], n_iters_total)

                    # measure elapsed time
                    batch_time = time.time() - end
                    end = time.time()

                    # dump to tensorboard per-iter time stats
                    writer.add_scalar(f"{name}/batch_time", batch_time, n_iters_total)
                    writer.add_scalar(f"{name}/data_time", data_time, n_iters_total)

                    # dump to tensorboard per-iter stats about sizes
                    writer.add_scalar(f"{name}/batch_size", batch_size, n_iters_total)
                    writer.add_scalar(f"{name}/n_views", n_views, n_iters_total)

                    n_iters_total += 1

            if DEBUG:
                print(f"Training of epoch {epoch}, batch {iter_i} complete!")

    # calculate evaluation metrics
    if master:
        if not is_train:
            if DEBUG:
                print("Calculating evaluation metrics... ", end="")

            results['keypoints_3d'] = np.concatenate(
                results['keypoints_3d'], axis=0)
            results['indexes'] = np.concatenate(results['indexes'])

            try:
                scalar_metric, full_metric = dataloader.dataset.evaluate(results['keypoints_3d'])
            except Exception as e:
                print("Failed to evaluate. Reason: ", e)
                scalar_metric, full_metric = 0.0, {}

            metric_dict['dataset_metric'].append(scalar_metric)

            checkpoint_dir = os.path.join(experiment_dir, "checkpoints", "{:04}".format(epoch))
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            if DEBUG:
                print("Calculated!")

            # dump results
            with open(os.path.join(checkpoint_dir, "results.pkl"), 'wb') as fout:
                if DEBUG:
                    print(f"Dumping results to {checkpoint_dir}/results.pkl... ", end="")
                pickle.dump(results, fout, protocol=4)
                if DEBUG:
                    print("Dumped!")

            # dump extra data as pkl file if need to reconstruct anything
            if save_extra_data: 
                with open(os.path.join(checkpoint_dir, "extra_data.pkl"), 'wb') as fout:
                    if DEBUG:
                        print(f"Dumping extra data to {checkpoint_dir}/extra_data.pkl... ", end="")

                    pickle.dump(extra_data, fout, protocol=4)
                    
                    if DEBUG:
                        print("Dumped!")

            # dump full metric
            with open(os.path.join(checkpoint_dir, "metric.json".format(epoch)), 'w') as fout:
                if DEBUG:
                    print(f"Dumping metric to {checkpoint_dir}/metric.json... ", end="")
                
                json.dump(full_metric, fout, indent=4, sort_keys=True)
                
                if DEBUG:
                    print("Dumped!")

        # dump to tensorboard per-epoch stats
        for title, value in metric_dict.items():
            writer.add_scalar(f"{name}/{title}_epoch", np.mean(value), epoch)

    print(f"Epoch {epoch} {train_eval_mode} complete!")

    return n_iters_total


def main(args):
    print("Number of available GPUs: {}".format(torch.cuda.device_count()))

    # Attempt to fix overflow error with pickle
    # See https://stackoverflow.com/questions/51562221/python-multiprocessing-overflowerrorcannot-serialize-a-bytes-object-larger-t
    ctx = torch.multiprocessing.get_context()
    ctx.reducer = pickle4reducer.Pickle4Reducer()

    config = cfg.load_config(args.config)

    global DEBUG
    DEBUG = config.debug_mode if hasattr(config, "debug_mode") else False
    print("Debugging Mode: ", DEBUG)

    is_distributed = False
    print("Using distributed:", is_distributed)

    master = True
    
    device = torch.device(0)

    # config
    config.opt.n_iters_per_epoch = config.opt.n_objects_per_epoch // config.opt.batch_size
    
    if hasattr(config.opt, "n_objects_per_epoch_val"):
        config.opt.n_iters_per_epoch_val = config.opt.n_objects_per_epoch_val // config.opt.val_batch_size
    else:
        config.opt.n_iters_per_epoch_val = None

    model = {
        "ransac": RANSACTriangulationNet,
        "alg": AlgebraicTriangulationNet,
        "vol": VolumetricTriangulationNet
    }[config.model.name](config)

    # NOTE: May be a bad idea to share memory since NCCL used
    # https://pytorch.org/docs/stable/distributed.html#torch.distributed.Backend
    # model.share_memory()

    if config.model.init_weights:
        state_dict = torch.load(config.model.checkpoint)
        for key in list(state_dict.keys()):
            new_key = key.replace("module.", "")
            state_dict[new_key] = state_dict.pop(key)

        model.load_state_dict(state_dict, strict=False)
        print("Successfully loaded pretrained weights for whole model")

    # criterion
    criterion_class = {
        "MSE": KeypointsMSELoss,
        "MSESmooth": KeypointsMSESmoothLoss,
        "MAE": KeypointsMAELoss
    }[config.opt.criterion]

    if config.opt.criterion == "MSESmooth":
        criterion = criterion_class(config.opt.mse_smooth_threshold)
    else:
        criterion = criterion_class()

    # optimizer
    opt = None
    if not args.eval:
        print("Optimising model...")
        if config.model.name == "vol":
            opt = torch.optim.Adam(
                [{'params': model.backbone.parameters()},
                 {'params': model.process_features.parameters(), 'lr': config.opt.process_features_lr if hasattr(config.opt, "process_features_lr") else config.opt.lr},
                 {'params': model.volume_net.parameters(), 'lr': config.opt.volume_net_lr if hasattr(config.opt, "volume_net_lr") else config.opt.lr}
                ],
                lr=config.opt.lr
            )
        else:
            opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.opt.lr)


    # datasets
    print("Loading data...")
    train_dataloader, val_dataloader, train_sampler = setup_dataloaders(config, distributed_train=is_distributed)

    # experiment
    experiment_dir, writer = None, None
    if master:
        experiment_dir, writer = setup_experiment(config, type(model).__name__, is_train=not args.eval)


    if not args.eval:
        print(f"Performing training with {config.opt.n_epochs} total epochs...")

        # train loop
        n_iters_total_train, n_iters_total_val = 0, 0
        for epoch in range(config.opt.n_epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            if DEBUG:
                print(f"Training epoch {epoch}...")

            # Cache needs to be emptied first
            # torch.cuda.empty_cache()
            # print("CUDA Cache Empty!")

            n_iters_total_train = one_epoch(model, criterion, opt, config, train_dataloader, epoch, n_iters_total=n_iters_total_train, is_train=True, master=master, experiment_dir=experiment_dir, writer=writer)

            if DEBUG:
                print(f"Epoch {epoch} training complete!")

                # torch.cuda.empty_cache()

                print(f"Evaluating epoch {epoch}...")

            n_iters_total_val = one_epoch(model, criterion, opt, config, val_dataloader, epoch, n_iters_total=n_iters_total_val, is_train=False, master=master, experiment_dir=experiment_dir, writer=writer)

            if DEBUG:
                print(f"Epoch {epoch} evaluation complete!")

            if master:
                checkpoint_dir = os.path.join(experiment_dir, "checkpoints", "{:04}".format(epoch))
                os.makedirs(checkpoint_dir, exist_ok=True)

                if DEBUG:
                    print(f"Saving checkpoints to {checkpoint_dir}/weights.pth... ", end="")

                torch.save(model.state_dict(), os.path.join(checkpoint_dir, "weights.pth"))

                if DEBUG:
                    print("Checkpoint saved!")

            print(f"{n_iters_total_train} iters done.")
    else:
        if args.eval_dataset == 'train':
            one_epoch(model, criterion, opt, config, train_dataloader, 0, n_iters_total=0, is_train=False, master=master, experiment_dir=experiment_dir, writer=writer)
        else:
            one_epoch(model, criterion, opt, config, val_dataloader, 0, n_iters_total=0, is_train=False, master=master, experiment_dir=experiment_dir, writer=writer)

    print("Done.")

if __name__ == '__main__':
    args = parse_args()
    print("args: {}".format(args))
    
    main(args)