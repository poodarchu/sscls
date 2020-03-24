#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Train a classification model."""

import argparse
import numpy as np
import os
import sys
import torch

import sscls.modeling.loss.builder as losses
import sscls.modeling.builder as model_builder
import sscls.solver.optimizer as optim
import sscls.datasets.loader as loader
import sscls.utils.benchmark as bu
import sscls.utils.checkpoint as cu
import sscls.utils.distributed as du
import sscls.utils.logging as lu
import sscls.utils.metrics as mu
import sscls.utils.multiprocessing as mpu
import sscls.utils.net as nu

from sscls.core.config import cfg, dump_cfg, assert_cfg
from sscls.utils.meters import TrainMeter, TestMeter

logger = lu.get_logger(__name__)


def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser(
        description='Train a classification model'
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
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def is_eval_epoch(cur_epoch):
    """Determines if the model should be evaluated at the current epoch."""
    return cfg.TRAIN.EVAL_PERIOD != -1 and (
        (cur_epoch + 1) % cfg.TRAIN.EVAL_PERIOD == 0 or
        (cur_epoch + 1) == cfg.OPTIM.MAX_EPOCH
    )


def log_model_info(model):
    """Logs model info"""
    logger.info('Model:\n{}'.format(model))
    logger.info('Params: {:,}'.format(mu.params_count(model)))
    logger.info('Flops: {:,}'.format(mu.flops_count(model)))


def train_epoch(
    cfg, train_loader, model, loss_fun, optimizer, train_meter, cur_epoch
):
    """Performs one epoch of training."""

    # Shuffle the data
    loader.shuffle(train_loader, cur_epoch)
    # Update the learning rate
    lr = optim.get_epoch_lr(cur_epoch)
    optim.set_lr(optimizer, lr)
    # Enable training mode
    model.train()
    train_meter.iter_tic()

    for cur_iter, (inputs, labels) in enumerate(train_loader):
        # Transfer the data to the current GPU device
        if cfg.USE_DPFLOW:
            inputs = torch.from_numpy(inputs)
            labels = torch.from_numpy(labels)

        if isinstance(inputs, list):
            inputs = [i.cuda(non_blocking=True) for i in inputs]
            labels = labels.cuda(non_blocking=True)
            input_size = inputs[0].size()[0]
        else:
            inputs, labels = inputs.cuda(), labels.cuda()
            input_size = inputs.size()[0]

        # Perform the forward pass
        preds = model(inputs)
        if cfg.MODEL.TYPE == 'moco':
            targets = preds[1]
            preds = preds[0]
        else:
            targets = labels
        # Compute the loss
        loss = loss_fun(preds, targets)
        # Perform the backward pass
        optimizer.zero_grad()
        loss.backward()
        # Update the parameters
        optimizer.step()

        top1_err, top5_err = mu.topk_errors(preds, targets, [1, 5])
        # # Combine the stats across the GPUs
        if cfg.NUM_GPUS > 1:
            loss, top1_err, top5_err = du.scaled_all_reduce(
                [loss, top1_err, top5_err]
            )
        # Copy the stats from GPU to CPU (sync point)
        top1_err, top5_err = top1_err.item(), top5_err.item()

        loss = loss.item()

        train_meter.iter_toc()
        # Update and log stats
        train_meter.update_stats(
            loss, lr, input_size * cfg.NUM_GPUS, top1_err, top5_err
        )
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()

    # Log epoch stats
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()


@torch.no_grad()
def test_epoch(cfg, test_loader, model, test_meter, cur_epoch):
    """Evaluates the model on the test set."""

    # Enable eval mode
    model.eval()
    test_meter.iter_tic()

    # for cur_iter, batch in enumerate(test_loader):
    for cur_iter, (inputs, labels) in enumerate(test_loader):
        if cfg.USE_DPFLOW:
            inputs = torch.from_numpy(inputs)
            labels = torch.from_numpy(labels)

        # Transfer the data to the current GPU device
        if isinstance(inputs, list):
            inputs = [i.cuda(non_blocking=True) for i in inputs]
            labels = labels.cuda(non_blocking=True)
            input_size = inputs[0].size(0)
        else:
            inputs, labels = inputs.cuda(), labels.cuda()
            input_size = inputs.size(0)

        # Compute the predictions
        preds = model(inputs)
        if cfg.MODEL.TYPE == 'moco':
            targets = preds[1]
            preds = preds[0]
        else:
            targets = labels
        # Compute the errors
        top1_err, top5_err = mu.topk_errors(preds, targets, [1, 5])
        # Combine the errors across the GPUs
        if cfg.NUM_GPUS > 1:
            top1_err, top5_err = du.scaled_all_reduce([top1_err, top5_err])
        # Copy the errors from GPU to CPU (sync point)
        top1_err, top5_err = top1_err.item(), top5_err.item()
        test_meter.iter_toc()
        # Update and log stats
        test_meter.update_stats(
            top1_err, top5_err, input_size * cfg.NUM_GPUS
        )
        test_meter.log_iter_stats(cur_epoch, cur_iter)
        test_meter.iter_tic()

    # Log epoch stats
    test_meter.log_epoch_stats(cur_epoch)
    test_meter.reset()


def train_model():
    """Trains the model."""

    # Build the model (before the loaders to speed up debugging)
    model = model_builder.build_model()
    log_model_info(model)

    # Define the loss function
    loss_fun = losses.get_loss_fun()
    # Construct the optimizer
    optimizer = optim.construct_optimizer(model)

    # Load checkpoint or initial weights
    start_epoch = 0
    if cfg.TRAIN.AUTO_RESUME and cu.has_checkpoint():
        last_checkpoint = cu.get_last_checkpoint()
        checkpoint_epoch = cu.load_checkpoint(last_checkpoint, model, optimizer)
        logger.info('Loaded checkpoint from: {}'.format(last_checkpoint))
        start_epoch = checkpoint_epoch + 1
    elif cfg.TRAIN.WEIGHTS:
        cu.load_checkpoint(cfg.TRAIN.WEIGHTS, model)
        logger.info('Loaded initial weights from: {}'.format(cfg.TRAIN.WEIGHTS))

    # Compute precise time
    if start_epoch == 0 and cfg.PREC_TIME.ENABLED:
        logger.info('Computing precise time...')
        bu.compute_precise_time(model, loss_fun)
        nu.reset_bn_stats(model)

    # Create data loaders
    train_loader = loader.construct_train_loader()
    test_loader = loader.construct_test_loader()

    # Create meters
    train_meter = TrainMeter(len(train_loader))
    test_meter = TestMeter(len(test_loader))

    # Perform the training loop
    logger.info('Start epoch: {}'.format(start_epoch + 1))

    for cur_epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):
        # Train for one epoch
        train_epoch(
            cfg, train_loader, model, loss_fun, optimizer, train_meter, cur_epoch
        )
        # Compute precise BN stats
        if cfg.BN.USE_PRECISE_STATS:
            nu.compute_precise_bn_stats(model, train_loader)
        # Save a checkpoint
        if cu.is_checkpoint_epoch(cur_epoch):
            checkpoint_file = cu.save_checkpoint(model, optimizer, cur_epoch)
            logger.info('Wrote checkpoint to: {}'.format(checkpoint_file))
        # Evaluate the model
        if is_eval_epoch(cur_epoch):
            test_epoch(cfg, test_loader, model, test_meter, cur_epoch)


def single_proc_train():
    """Performs single process training."""

    # Setup logging
    lu.setup_logging()
    # Show the config
    logger.info('Config:\n{}'.format(cfg))

    # Fix the RNG seeds (see RNG comment in core/config.py for discussion)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    # Configure the CUDNN backend
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK

    # Train the model
    train_model()


def main():
    # Parse cmd line args
    args = parse_args()

    # Load config options
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    assert_cfg()
    cfg.freeze()

    # Ensure that the output dir exists
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    # Save the config
    dump_cfg()

    # Perform training
    if cfg.NUM_GPUS > 1:
        mpu.multi_proc_run(num_proc=cfg.NUM_GPUS, fun=single_proc_train)
    else:
        single_proc_train()


if __name__ == '__main__':
    main()
