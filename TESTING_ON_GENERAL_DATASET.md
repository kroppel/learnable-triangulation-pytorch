# Testing on a General Dataset

This document aims to outline how to use the code in the repository herein on a general dataset (could be your own). For examples, see the additional code and documentation for the addition of the [CMU Panoptic Studio](http://domedb.perception.cs.cmu.edu/) dataset at [mvn/datasets/cmu_preprocessing/README.md](https://github.com/Samleo8/learnable-triangulation-pytorch/blob/master/mvn/datasets/cmu_preprocessing/README.md)

Note that this document only entails **testing** on a general dataset. If you would like to use your dataset for training, you can consult the other document [here](TRAINING_ON_GENERAL_DATASET.md).

## Overview

There are actually 2 main parts that are required before to fully do testing/training:

1. Generate a labels file (`.npy`) file containing all the necessary data the algorithm needs, as listed in the [requirements](#requirements) section below. This is done using a `generate labels.py` file, specific to the dataset and [how the data is organised](#data-organisation).
2. A subclass of the pytorch `Dataset` class that loads information specific to your dataset, as organised in your npy labels file.

## Requirements

For testing, you will need the following data:

- Images (not videos) for each frame
- Camera extrinsics: `R, t, K, dist_coefficient` components of camera matrix for each camera
- Bounding boxes for individual persons *(see (Generating bounding boxes)[#generating-bounding-boxes])*
- *Ground truth/predicted pelvis positions (optional, for volumetric only)*

For training, see [these requirements](TRAINING_ON_GENERAL_DATASET.md#requirements)

## Data Organisation

Preferably, 

## Generating bounding boxes

WIP
