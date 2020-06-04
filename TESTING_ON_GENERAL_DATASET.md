# Testing on a General Dataset

This document aims to outline how to use the code in the repository herein on a general dataset (could be your own). For examples, see the additional code and documentation for the addition of the [CMU Panoptic Studio](http://domedb.perception.cs.cmu.edu/) dataset at [mvn/datasets/cmu_preprocessing/README.md](https://github.com/Samleo8/learnable-triangulation-pytorch/blob/master/mvn/datasets/cmu_preprocessing/README.md)

Note that this document only entails **testing** on a general dataset. If you would like to use your dataset for training, you can consult the other document [here](TRAINING_ON_GENERAL_DATASET.md).

## Overview

There are actually 3 main parts that one is required to do before to fully do testing/training:

1. Generate a labels file (`.npy`) file containing all the necessary data the algorithm needs, as listed in the [requirements](#requirements) section below. This is done using a `generate labels.py` file under `mvn/datasets/<your_dataset>`, specific to the dataset and [how the data is organised](#data-organisation). 

   Part of this label file generation will also include generating a consolidated `npy` file with the BBOX data. This may be done separately using another python script.

2. Create a subclass of the pytorch `Dataset` class that loads information specific to your dataset, as organised in your npy labels file. This should be in `mvn/datasets/`
3. Create config files under the `experiments` folder that tell the algorithm how to handle your data.

- [Testing on a General Dataset](#testing-on-a-general-dataset)
  - [Overview](#overview)
- [1. Generating the Labels](#1-generating-the-labels)
  - [Requirements](#requirements)
  - [Data Organisation](#data-organisation)
    - [Images](#images)
    - [Camera Calibration Data](#camera-calibration-data)
    - [Pose Data (needed for training, and volumetric triangulation testing)](#pose-data-needed-for-training-and-volumetric-triangulation-testing)
  - [Generating bounding boxes](#generating-bounding-boxes)
    - [Algorithm for BBOXes](#algorithm-for-bboxes)
    - [BBOX Labels File](#bbox-labels-file)
  - [Needed Python Scripts](#needed-python-scripts)
    - [Labels Generation Script](#labels-generation-script)
    - [BBOX Generation Script](#bbox-generation-script)

# 1. Generating the Labels

## Requirements

For testing (and training), you will need the following data:

## Data Organisation

Preferably, the data should be organised similar to that of the CMU Panoptic Studio dataset, where the data is grouped by `action/scene` > `camera` > `person`.

Specifically, it would be good if the data is organised as below. Of course, the data does not necessarily have to be in the exact format; you would just need to make the appropriate changes to the respective [label generation](#labels-generation-script) and dataset subclass files.

### Images

`$DIR_ROOT/[ACTION_NAME]/hdImgs/[CAMERA_ID]/[FRAME_ID].jpg`

### Camera Calibration Data

`$DIR_ROOT/[ACTION_NAME]/calibration_[ACTION_NAME].json`

The JSON data should have this format, with the camera IDs in their appropriate order, or labelled accordingly:

```json
[
    {
        'id':   0 // optional
        'R':    [3x3 rotation matrix],
        'k':    [3x3 calibration/instrinsics matrix]
        't':    [3x1 translation matrix]
        'dist': [5x1 distortion coefficients]
    },
    {
        'id':   1 // optional
        'R':    [3x3 rotation matrix],
        'k':    [3x3 calibration/instrinsics matrix]
        't':    [3x1 translation matrix]
        'dist': [5x1 distortion coefficients]
    },
    {
        ...
    }
]
```

More information on distortion coefficients [here](#https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html).

### Pose Data (needed for training, and volumetric triangulation testing)

`$DIR_ROOT/[ACTION_NAME]/3DKeypoints_[FRAME_ID].json`

The JSON data should have this notable format:

```json
[
    {
        'id':     [PERSON_ID],
        'joints': [ARRAY OF JOINT COORDINATES IN COCO 19 FORMAT]
    },
    {
        ...
    }
]
```

## Generating bounding boxes

There are 2 inherent parts to this: the algorithm (MRCNN or SSD) to figure out the actual bounding boxes; and the generation of a labels file that consolidates the said data into a single labels file.

### Algorithm for BBOXes

This repository does not contain any algorithm to detect persons in the scene. For now, you need to find your own. Popular algorithms include Mask-RCNN (MRCNN) and Single Shot Detectors (SSD). Current SOTA include [Detectron2](https://github.com/facebookresearch/detectron2) and [MM Detection](https://github.com/open-mmlab/mmdetection).

The data should ideally be organised by **action/scene** > **camera ID** > **person ID** with a JSON file containing an array of BBOXes in order of frame number.

### BBOX Labels File

A python script is needed to consolidate the bouding box labels. More information is given [below](#bbox-generation-script)

## Needed Python Scripts

### Labels Generation Script

### BBOX Generation Script
