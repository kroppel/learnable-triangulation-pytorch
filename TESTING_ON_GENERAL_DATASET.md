# Testing on a General Dataset

This document aims to outline how to use the code in the repository herein on a general dataset (could be your own). For examples, see the additional code and documentation for the addition of the [CMU Panoptic Studio](http://domedb.perception.cs.cmu.edu/) dataset at [mvn/datasets/cmu_preprocessing/README.md](https://github.com/Samleo8/learnable-triangulation-pytorch/blob/master/mvn/datasets/cmu_preprocessing/README.md)

Note that this document only entails **testing** on a general dataset. If you would like to use your dataset for training, you can consult the other document [here](TRAINING_ON_GENERAL_DATASET.md).

# Requirements
For testing, you will need the following data:

- Images (not videos) for each frame
- Camera extrinsics: `R, t, K, dist_coefficient` components of camera matrix for each camera
- Bounding boxes for individual persons *(see (Generating bounding boxes)[#generating-bounding-boxes])*
- *Ground truth/predicted pelvis positions (optional, for volumetric only)*

For training, see [these requirements](TRAINING_ON_GENERAL_DATASET.md#requirements)

## Generating bounding boxes
