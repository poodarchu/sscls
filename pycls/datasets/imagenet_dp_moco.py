#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""ImageNet dataset."""

import numpy as np
import torch
import torch.utils.data
import torchvision.transforms as torch_transforms
import nori2 as nori

from PIL import Image
from brainpp.oss import OSSPath

from pycls.core.config import cfg
from pycls.models import moco
from pycls.datasets.utils import imgproc

import pycls.datasets.transforms as transforms
import pycls.utils.logging as lu

logger = lu.get_logger(__name__)

# Per-channel mean and SD values in BGR order
_MEAN = [0.406, 0.456, 0.485]
_SD = [0.225, 0.224, 0.229]

# Eig vals and vecs of the cov mat
_EIG_VALS = np.array([
    [0.2175, 0.0188, 0.0045]
])
_EIG_VECS = np.array([
    [-0.5675, 0.7192, 0.4009],
    [-0.5808, -0.0045, -0.8140],
    [-0.5836, -0.6948, 0.4203]
])


class ImageNetDPMoCo(torch.utils.data.Dataset):
    """ImageNet dataset."""

    def __init__(self, data_path, split):
        self._data_path = data_path
        self._split = split
        logger.info(
            f'Constructing ImageNet {self._split} from path {self._data_path[self._split]}...')
        self.transform = self._prepare_im
        self.nid_filename = self._data_path[self._split]
        self.fetcher = None

        self.nid_labels = []
        if "://" in self.nid_filename:
            f = OSSPath(self.nid_filename).open("r")
        else:
            f = open(self.nid_filename, "r")
        for line in f.readlines():
            nid, label, _ = line.strip().split("\t")
            self.nid_labels.append((nid, int(label)))
        f.close()

        # transforms
        normalize = torch_transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if cfg.MOCO.V2_AUG:
            augmentation = [
                torch_transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                torch_transforms.RandomApply([
                    torch_transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                torch_transforms.RandomGrayscale(p=0.2),
                torch_transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
                torch_transforms.RandomHorizontalFlip(),
                torch_transforms.ToTensor(),
                normalize
            ]
        else:
            augmentation = [
                torch_transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                torch_transforms.RandomGrayscale(p=0.2),
                torch_transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                torch_transforms.RandomHorizontalFlip(),
                torch_transforms.ToTensor(),
                normalize
            ]

        self.transform = moco.loader.TwoCropsTransform(
            torch_transforms.Compose(augmentation))

    def _prepare_im(self, im):
        """Prepares the image for network input."""
        # Train and test setups differ
        if self._split == 'train':
            im = self.transform(Image.fromarray(im))
        else:
            # Scale and center crop
            im = transforms.scale(cfg.TEST.IM_SIZE, im)
            im = transforms.center_crop(cfg.TRAIN.IM_SIZE, im)
            # HWC -> CHW
            im = im.transpose([2, 0, 1])
            # [0, 255] -> [0, 1]
            im = im / 255.0
            # PCA jitter
            if self._split == 'train':
                im = transforms.lighting(im, 0.1, _EIG_VALS, _EIG_VECS)
            # Color normalization
            im = transforms.color_norm(im, _MEAN, _SD)

        return im

    def __getitem__(self, index):
        if not self.fetcher:
            self.fetcher = nori.Fetcher()
        nid, label = self.nid_labels[index]
        img = imgproc.imdecode(self.fetcher.get(nid))[..., :3]
        imgs = self._prepare_im(img)
        # return [image.numpy().astype(np.float32) for image in imgs], label
        return imgs, label

    def __len__(self):
        return len(self.nid_labels)
