# sscls 

**sscls** is an image classification codebase for research related to self-supervised representation learning, written in [PyTorch](https://pytorch.org/). sscls is based on [pycls](https://github.com/facebookresearch/pycls).

## Introduction

The goal of **sscls** is to provide a high-quality, high-performance codebase for self-supervised <i>research</i>. It is designed to be <i>simple</i> and <i>flexible</i> in order to support rapid implementation and evaluation of research ideas.

The codebase implements efficient single-machine multi-gpu training, powered by PyTorch distributed package. **sscls** includes implementations of standard baseline models ([ResNet](https://arxiv.org/abs/1512.03385), [ResNeXt](https://arxiv.org/abs/1611.05431), [EfficientNet](https://arxiv.org/abs/1905.11946)) and generic modeling functionality that can be useful for experimenting with network design. Additional models can be easily implemented. Apart from basic classification models, it has also included state-of-the-art [MoCo](https://arxiv.org/abs/1911.05722) v1&v2.

## Installation

Please see [`INSTALL.md`](docs/INSTALL.md) for installation instructions.

## Getting Started

After installation, please see [`GETTING_STARTED.md`](docs/GETTING_STARTED.md) for basic instructions on training and evaluation with **pycls**.

## Model Zoo
* resnet-50 epoch: 100, top1_err: 23.484001, top5_err: 6.814002

## License

**sscls** is released under the MIT license. See the [LICENSE](LICENSE) file for more information.
