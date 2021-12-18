# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
A module with common functions.
"""
import os
import torch
import numpy as np


def get_chk_name(name, fold, epoch):
    return "{name}_{fold}_epoch_{epoch}.pth".format(name=name or "model",fold=fold,epoch=epoch)


def checkpoint(model, epoch, fold, outdir, name=None, optimizer=None, scheduler=None,
               **kwargs):
    """ Save the weights of a given model.

    Parameters
    ----------
    model: Net
        the network model.
    epoch: int
        the epoch index.
    fold: int
        the fold index.
    outdir: str
        the destination directory where a 'model_<fold>_epoch_<epoch>.pth'
        file will be generated.
    optimizer: Optimizer, default None
        the network optimizer (save the hyperparameters, etc.).
    scheduler: Scheduler, default None
        the network scheduler.
    kwargs: dict
        others parameters to save.
    """

    name = get_chk_name(name, fold, epoch)
    outfile = os.path.join(
        outdir, name)
    if optimizer is not None:
        kwargs.update(optimizer=optimizer.state_dict())
    if scheduler is not None:
        kwargs.update(scheduler=scheduler.state_dict())
    torch.save({
        "fold": fold,
        "epoch": epoch,
        "model": model.state_dict(),
        **kwargs}, outfile)
    return outfile

def reset_weights(model, checkpoint=None):
    """ Reset all the weights of a model. If a checkpoint is given, restore
    the checkpoint weights.

    Parameters
    ----------
    model: Net
        the network model.
    checkpoint: dict
        the saved model weights
    """
    def weight_reset(m):
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
    if checkpoint is None:
        model.apply(weight_reset)
    else:
        if hasattr(checkpoint, "state_dict"):
            model.load_state_dict(checkpoint.state_dict())
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint)

def tensor2im(tensor):
    """
    It returns a numpy array from an input tensor which can share the memory with the input
    """
    if not isinstance(tensor, np.ndarray):
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
    return tensor

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

