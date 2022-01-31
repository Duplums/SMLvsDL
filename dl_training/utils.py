# -*- coding: utf-8 -*-
"""
A module with common functions.
"""
import os
import torch
import numpy as np
import logging
import warnings

logger = logging.getLogger("SMLvsDL")

def get_chk_name(name, fold, epoch):
    return "{name}_{fold}_epoch_{epoch}.pth".format(name=name or "model",fold=fold,epoch=epoch)

def get_pickle_obj(path):
    import pickle
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj

def setup_logging(level="info", logfile=None):
    """ Setup the logging.

    Parameters
    ----------
    logfile: str, default None
        the log file.
    """
    LEVELS = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL
    }
    logging_format = logging.Formatter(
        "[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - "
        "%(message)s", "%Y-%m-%d %H:%M:%S")
    while len(logging.root.handlers) > 0:
        logging.root.removeHandler(logging.root.handlers[-1])
    while len(logger.handlers) > 0:
        logger.removeHandler(logger.handlers[-1])
    level = LEVELS.get(level, None)
    if level is None:
        raise ValueError("Unknown logging level.")
    logger.setLevel(level)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(logging_format)
    logger.addHandler(stream_handler)
    if logfile is not None:
        file_handler = logging.FileHandler(logfile, mode="a")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging_format)
        logger.addHandler(file_handler)
    if level != logging.DEBUG:
        warnings.simplefilter("ignore", DeprecationWarning)

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

