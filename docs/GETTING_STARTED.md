# Getting Started

This document provides basic instructions for training and evaluation using **sscls**.

- For general information about **sscls**, please see [`README.md`](../README.md)
- For installation instructions, please see [`INSTALL.md`](INSTALL.md)

## Training Models

Training on CIFAR with 1 GPU: 

```
python tools/train_net.py \
    --cfg configs/cifar/resnet/R-56_nds_1gpu.yaml \
    OUT_DIR /tmp
```

Training on ImageNet with 8 GPU:

```
python tools/train_net.py \
    --cfg configs/imagenet/resnet/R-50-1x64d_step_8gpu_dpflow.yaml \
    OUT_DIR /tmp
```

Training MoCo on ImageNet with 8 GPU:
```
python tools/train_net.py \
    --cfg configs/imagenet/moco/MoCo_R-50-1x64d_step_8gpu_dpflow.yaml \
    OUTPUT_DIR /tmp
```
Training MoCo v2 on ImageNet with 8 GPU:
```
python tools/train_net.py \
    --cfg configs/imagenet/moco/MoCov2_R-50-1x64d_step_8gpu_dpflow.yaml \
    OUTPUT_DIR /tmp
```

## Finetuning Models

Finetuning on ImageNet with 8 GPU:

```
python tools/train_net.py \
    --cfg configs/imagenet/resnet/R-50-1x64d_step_8gpu.yaml \
    TRAIN.WEIGHTS /path/to/weights/file \
    OUT_DIR /tmp
```

## Evaluating Models

Evaluation on ImageNet with 8 GPU:

```
python tools/test_net.py \
    --cfg configs/imagenet/resnet/R-50-1x64d_step_8gpu.yaml \
    TEST.WEIGHTS /path/to/weights/file \
    OUT_DIR /tmp
```
