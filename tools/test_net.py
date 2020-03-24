#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test a trained classification model."""

import argparse
import numpy as np
import sys
import torch

from sscls.core.config import assert_cfg
from sscls.core.config import cfg
from sscls.utils.meters import TestMeter

import sscls.modeling.builder as model_builder
import sscls.datasets.loader as loader
import sscls.utils.checkpoint as cu
import sscls.utils.distributed as du
import sscls.utils.logging as lu
import sscls.utils.metrics as mu
import sscls.utils.multiprocessing as mpu

logger = lu.get_logger(__name__)


def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser(
        description='Test a trained classification model'
    )
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='Config file',
        required=True,
        type=str
    )
    parser.add_argument(
        'opts',
        help='See sscls/core/config/defaults.py for all options',
        default=None,
        nargs=argparse.REMAINDER
    )
    if len(sys.argv) == 1:
        parser.logger.info_help()
        sys.exit(1)
    return parser.parse_args()


def log_model_info(model):
    """Logs model info"""
    logger.info('Model:\n{}'.format(model))
    logger.info('Params: {:,}'.format(mu.params_count(model)))
    logger.info('Flops: {:,}'.format(mu.flops_count(model)))


@torch.no_grad()
def test_epoch(test_loader, model, test_meter, cur_epoch):
    """Evaluates the model on the test set."""

    # Enable eval mode
    model.eval()
    test_meter.iter_tic()

    for cur_iter, (inputs, labels) in enumerate(test_loader):
        # Transfer the data to the current GPU device
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        # Compute the predictions
        preds = model(inputs)
        # Compute the errors
        top1_err, top5_err = mu.topk_errors(preds, labels, [1, 5])
        # Combine the errors across the GPUs
        if cfg.NUM_GPUS > 1:
            top1_err, top5_err = du.scaled_all_reduce([top1_err, top5_err])
        # Copy the errors from GPU to CPU (sync point)
        top1_err, top5_err = top1_err.item(), top5_err.item()
        test_meter.iter_toc()
        # Update and log stats
        test_meter.update_stats(
            top1_err, top5_err, inputs.size(0) * cfg.NUM_GPUS
        )
        test_meter.log_iter_stats(cur_epoch, cur_iter)
        test_meter.iter_tic()

    # Log epoch stats
    test_meter.log_epoch_stats(cur_epoch)
    test_meter.reset()


def test_model():
    """Evaluates the model."""

    # Build the model (before the loaders to speed up debugging)
    model = model_builder.build_model()
    log_model_info(model)

    # Load model weights
    cu.load_checkpoint(cfg.TEST.WEIGHTS, model)
    logger.info('Loaded model weights from: {}'.format(cfg.TEST.WEIGHTS))

    # Create data loaders
    test_loader = loader.construct_test_loader()

    # Create meters
    test_meter = TestMeter(len(test_loader))

    # Evaluate the model
    test_epoch(test_loader, model, test_meter, 0)


def single_proc_test():
    """Performs single process evaluation."""

    # Setup logging
    lu.setup_logging()
    # Show the config
    logger.info('Config:\n{}'.format(cfg))

    # Fix the RNG seeds (see RNG comment in core/config.py for discussion)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    # Configure the CUDNN backend
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK

    # Evaluate the model
    test_model()


def main():
    # Parse cmd line args
    args = parse_args()

    # Load config options
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    assert_cfg()
    cfg.freeze()

    # Perform evaluation
    if cfg.NUM_GPUS > 1:
        mpu.multi_proc_run(num_proc=cfg.NUM_GPUS, fun=single_proc_test)
    else:
        single_proc_test()


if __name__ == '__main__':
    main()
