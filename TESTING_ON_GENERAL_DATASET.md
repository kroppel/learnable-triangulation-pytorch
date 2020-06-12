# Testing on a General Dataset

Before we begin, make sure you have followed the steps [here](SETUP_GENERAL_DATASET.md) to setup your dataset.

This README aims to document the additional steps required to **test/evaluate** a general dataset. If you would like to evaluate it instead, see the document for [training a general dataset](TRAINING_ON_GENERAL_DATASET.md).

Note that there are 2 types of "testing" that you can do:

1. [Evaluation](#evaluation): you have ground truth keypoints data, and what to evaluate against this ground truth
2. [Demo](#demo): you do not have ground truth keypoints, and want to use this algorithm to generate these keypoints

## Evaluation

Simply run

```bash
python3 train.py \
  --eval --eval_dataset val \
  --config experiments/path/to/config_file.yaml \
  --logdir ./logs
```

Argument `--eval_dataset` can be `val` or `train`. Results can be seen in `logs` directory or in the tensorboard.

## Demo

Simply run

```bash
python3 demo.py \
  --config experiments/path/to/config_file.yaml \
  --logdir ./logs
```

Argument `--eval_dataset` can be `val` or `train`. Results can be seen in `logs` directory or in the tensorboard.

## Visualising Results

To visualise your results, follow the instructions in the [README](README.md). You can choose to run [with](README.md#tensorboard
) or [without]((README.md#visualising-results-without-tensorboard) tensorboard.
