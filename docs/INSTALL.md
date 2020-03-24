# Installation Instructions

This document covers how to install **sscls** and its dependencies.

- For general information about **sscls**, please see [`README.md`](../README.md)

**Requirements:**

- NVIDIA GPU, Linux, Python3
- PyTorch, various Python packages; Instructions for installing these dependencies are found below

**Notes:**

- **sscls** does not currently support running on CPU; a GPU system is required
- **sscls** has been tested with CUDA 10.1 and cuDNN 7.6

## PyTorch

To install PyTorch with CUDA support, follow the [installation instructions](https://pytorch.org/get-started/locally/) from the [PyTorch website](https://pytorch.org).

## sscls

Clone the **sscls** repository:

```
git clone https://github.com/poodarchu/sscls 
```

Install Python dependencies:

```
pip install -r requirements.txt
```

Set up Python modules:

```
python setup.py build develop
```

## Datasets

**sscls** finds datasets via symlinks from `sscls/datasets/data` to the actual locations where the dataset images and annotations are stored. For instructions on how to create symlinks for CIFAR and ImageNet, please see [`DATA.md`](DATA.md). Apart from basic folder dataset, we also add DPFlow and Nori support, such as [ImageNetDP](sscls/datasets/imagenet_dp.py)

## Getting Started

Please see [`GETTING_STARTED.md`](GETTING_STARTED.md) for basic instructions on training and evaluation with **sscls**.
