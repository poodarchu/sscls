#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Data loader."""

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

import torch

from sscls.core.config import cfg
from sscls.datasets.cifar10 import Cifar10
from sscls.datasets.imagenet import ImageNet
from sscls.datasets.imagenet_dp import ImageNetDP
from sscls.datasets.imagenet_dp_moco import ImageNetDPMoCo
from sscls.datasets.custom_loader import DPFlowDataLoader

import sscls.datasets.paths as dp

# Supported datasets
_DATASET_CATALOG = {
    'cifar10': Cifar10,
    'imagenet': ImageNet,
    'imagenet_dpflow': ImageNetDP,
    'imagenet_dpflow_moco': ImageNetDPMoCo,
}


def _construct_loader(dataset_name, split, batch_size, shuffle, drop_last):
    """Constructs the data loader for the given dataset."""
    assert dataset_name in _DATASET_CATALOG.keys(), \
        'Dataset \'{}\' not supported'.format(dataset_name)
    assert dp.has_data_path(dataset_name), \
        'Dataset \'{}\' has no data path'.format(dataset_name)
    # Retrieve the data path for the dataset
    data_path = dp.get_data_path(dataset_name)
    # Construct the dataset
    dataset = _DATASET_CATALOG[dataset_name](data_path, split)
    # Create a sampler for multi-process training
    sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None
    # Create a loader
    if not cfg.USE_DPFLOW:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(False if sampler else shuffle),
            sampler=sampler,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
            drop_last=drop_last,
        )
    else:
        loader = DPFlowDataLoader(
            dataset,
            dataset_name=dataset_name,
            batch_size=batch_size * cfg.NUM_GPUS,
            nr_gpu=cfg.NUM_GPUS,
            num_machines=cfg.NUM_REPLICAS,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,
            preemptible=False
        )
    return loader


def construct_train_loader():
    """Train loader wrapper."""
    return _construct_loader(
        dataset_name=cfg.TRAIN.DATASET,
        split=cfg.TRAIN.SPLIT,
        batch_size=int(cfg.TRAIN.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=True,
        drop_last=True
    )


def construct_test_loader():
    """Test loader wrapper."""
    return _construct_loader(
        dataset_name=cfg.TEST.DATASET,
        split=cfg.TEST.SPLIT,
        batch_size=int(cfg.TEST.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=False,
        drop_last=False
    )


def shuffle(loader, cur_epoch):
    if isinstance(loader, torch.utils.data.DataLoader):
        """"Shuffles the data."""
        assert isinstance(loader.sampler, (RandomSampler, DistributedSampler)), \
            'Sampler type \'{}\' not supported'.format(type(loader.sampler))
        # RandomSampler handles shuffling automatically
        if isinstance(loader.sampler, DistributedSampler):
            # DistributedSampler shuffles data based on epoch
            loader.sampler.set_epoch(cur_epoch)
