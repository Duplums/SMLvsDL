# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Core classes.
"""

# System import
import os
import pickle
from copy import deepcopy
import subprocess
# Third party import
import torch
import torch.nn.functional as func
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
# Package import
from dl_training.utils import checkpoint
from dl_training.history import History
import dl_training.metrics as mmetrics
import logging

class Base(object):
    """ Class to perform classification.
    """
    def __init__(self, optimizer_name="Adam", learning_rate=1e-3,
                 loss_name="NLLLoss", metrics=None, use_cuda=False,
                 pretrained=None, load_optimizer=True, use_multi_gpu=True,
                 **kwargs):
        """ Class instantiation.

        Observers will be notified, allowed signals are:
        - 'before_epoch'
        - 'after_epoch'

        Parameters
        ----------
        optimizer_name: str, default 'Adam'
            the name of the optimizer: see 'torch.optim' for a description
            of available optimizer.
        learning_rate: float, default 1e-3
            the optimizer learning rate.
        loss_name: str, default 'NLLLoss'
            the name of the loss: see 'torch.nn' for a description
            of available loss.
        metrics: list of str
            a list of extra metrics that will be computed.
        use_cuda: bool, default False
            whether to use GPU or CPU.
        pretrained: path, default None
            path to the pretrained model or weights.
        load_optimizer: boolean, default True
            if pretrained is set, whether to also load the optimizer's weights or not
        use_multi_gpu: boolean, default True
            if several GPUs are available, use them during forward/backward pass
        kwargs: dict
            specify directly a custom 'model', 'optimizer' or 'loss'. Can also
            be used to set specific optimizer parameters.
        """
        self.optimizer = kwargs.get("optimizer")
        self.logger = logging.getLogger("SMLvsDL")
        self.loss = kwargs.get("loss")
        self.device = torch.device("cuda" if use_cuda else "cpu")
        for name in ("optimizer", "loss"):
            if name in kwargs:
                kwargs.pop(name)
        if "model" in kwargs:
            self.model = kwargs.pop("model")
        if self.optimizer is None:
            if optimizer_name in dir(torch.optim):
                self.optimizer = getattr(torch.optim, optimizer_name)(
                    self.model.parameters(),
                    lr=learning_rate,
                    **kwargs)
            else:
                raise ValueError("Optimizer '{0}' uknown: check available "
                                 "optimizer in 'pytorch.optim'.")
        if self.loss is None:
            if loss_name not in dir(torch.nn):
                raise ValueError("Loss '{0}' uknown: check available loss in "
                                 "'pytorch.nn'.")
            self.loss = getattr(torch.nn, loss_name)()
        self.metrics = {}
        for name in (metrics or []):
            if name not in mmetrics.METRICS:
                raise ValueError("Metric '{0}' not yet supported: you can try "
                                 "to fill the 'METRICS' factory, or ask for "
                                 "some help!".format(name))
            self.metrics[name] = mmetrics.METRICS[name]
        if use_cuda and not torch.cuda.is_available():
            raise ValueError("No GPU found: unset 'use_cuda' parameter.")
        if pretrained is not None:
            checkpoint = None
            try:
                checkpoint = torch.load(pretrained, map_location=lambda storage, loc: storage)
            except BaseException as e:
                self.logger.error('Impossible to load the checkpoint: %s' % str(e))
            if checkpoint is not None:
                if hasattr(checkpoint, "state_dict"):
                    self.model.load_state_dict(checkpoint.state_dict())
                elif isinstance(checkpoint, dict):
                    if "model" in checkpoint:
                        try:
                            for key in list(checkpoint['model'].keys()):
                                if key.replace('module.', '') != key:
                                    checkpoint['model'][key.replace('module.', '')] = checkpoint['model'][key]
                                    del(checkpoint['model'][key])
                            #####
                            unexpected= self.model.load_state_dict(checkpoint["model"], strict=False)
                            self.logger.info('Model loading info: {}'.format(unexpected))
                            self.logger.info('Model loaded')
                        except BaseException as e:
                            self.logger.error('Error while loading the model\'s weights: %s' % str(e))
                            raise ValueError("")
                    if "optimizer" in checkpoint:
                        if load_optimizer:
                            try:
                                self.optimizer.load_state_dict(checkpoint["optimizer"])
                                for state in self.optimizer.state.values():
                                    for k, v in state.items():
                                        if torch.is_tensor(v):
                                            state[k] = v.to(self.device)
                            except BaseException as e:
                                self.logger.error('Error while loading the optimizer\'s weights: %s' % str(e))
                        else:
                            self.logger.warning("The optimizer's weights are not restored ! ")
                else:
                    self.model.load_state_dict(checkpoint)

        if use_multi_gpu and torch.cuda.device_count() > 1:
            self.model = DataParallel(self.model)

        self.model = self.model.to(self.device)

    def training(self, manager, nb_epochs: int, checkpointdir=None,
                 fold_index=None, scheduler=None, with_validation=True,
                 nb_epochs_per_saving=1, exp_name=None, **kwargs_train):
        """ Train the model.

        Parameters
        ----------
        manager: a dl_training DataManager
            a manager containing the train and validation data.
        nb_epochs: int, default 100
            the number of epochs.
        checkpointdir: str, default None
            a destination folder where intermediate models/historues will be
            saved.
        fold_index: int or [int] default None
            the index(es) of the fold(s) to use for the training, default use all the
            available folds.
        scheduler: torch.optim.lr_scheduler, default None
            a scheduler used to reduce the learning rate.
        with_validation: bool, default True
            if set use the validation dataset.
        nb_epochs_per_saving: int, default 1,
            the number of epochs after which the model+optimizer's parameters are saved
        exp_name: str, default None
            the experience name that will be launched
        Returns
        -------
        train_history, valid_history: History
            the train/validation history.
        """

        train_history = History(name="Train_%s"%(exp_name or ""))
        if with_validation is not None:
            valid_history = History(name="Validation_%s"%(exp_name or ""))
        else:
            valid_history = None
        print(self.loss)
        print(self.optimizer)
        folds = range(manager.get_nb_folds())
        if fold_index is not None:
            if isinstance(fold_index, int):
                folds = [fold_index]
            elif isinstance(fold_index, list):
                folds = fold_index
        init_optim_state = deepcopy(self.optimizer.state_dict())
        init_model_state = deepcopy(self.model.state_dict())
        if scheduler is not None:
            init_scheduler_state = deepcopy(scheduler.state_dict())
        for fold in folds:
            # Initialize everything before optimizing on a new fold
            self.optimizer.load_state_dict(init_optim_state)
            self.model.load_state_dict(init_model_state)
            if scheduler is not None:
                scheduler.load_state_dict(init_scheduler_state)
            loader = manager.get_dataloader(
                train=True,
                validation=True,
                fold_index=fold)
            for epoch in range(nb_epochs):
                loss, values = self.train(loader.train, fold, epoch, **kwargs_train)

                train_history.log((fold, epoch), loss=loss, **values)
                train_history.summary()
                if scheduler is not None:
                    scheduler.step()
                    print('Scheduler lr: {}'.format(scheduler.get_lr()), flush=True)
                    print('Optimizer lr: %f'%self.optimizer.param_groups[0]['lr'], flush=True)
                if checkpointdir is not None and (epoch % nb_epochs_per_saving == 0 or epoch == nb_epochs-1) \
                        and epoch > 0:
                    if not os.path.isdir(checkpointdir):
                        subprocess.check_call(['mkdir', '-p', checkpointdir])
                        self.logger.info("Directory %s created."%checkpointdir)
                    checkpoint(
                        model=self.model,
                        epoch=epoch,
                        fold=fold,
                        outdir=checkpointdir,
                        name=exp_name,
                        optimizer=self.optimizer)
                    train_history.save(
                        outdir=checkpointdir,
                        epoch=epoch,
                        fold=fold)
                if with_validation:
                    _, _, _, loss, values = self.test(loader.validation, **kwargs_train)
                    valid_history.log((fold, epoch), validation_loss=loss, **values)
                    valid_history.summary()
                    if checkpointdir is not None and (epoch % nb_epochs_per_saving == 0 or epoch == nb_epochs-1) \
                            and epoch > 0:
                        valid_history.save(
                            outdir=checkpointdir,
                            epoch=epoch,
                            fold=fold)
        return train_history, valid_history

    def train(self, loader,fold=None, epoch=None, **kwargs):
        """ Train the model on the trained data.

        Parameters
        ----------
        loader: a pytorch Dataloader

        Returns
        -------
        loss: float
            the value of the loss function.
        values: dict
            the values of the metrics.
        """

        self.model.train()
        nb_batch = len(loader)
        pbar = tqdm(total=nb_batch, desc="Mini-Batch")

        values = {}
        iteration = 0
        losses = []
        y_pred = []
        y_true = []
        for dataitem in loader:
            pbar.update()
            inputs = dataitem.inputs
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.to(self.device)
            list_targets = []
            _targets = []
            for item in (dataitem.outputs, dataitem.labels):
                if item is not None:
                    _targets.append(item.to(self.device))
            if len(_targets) == 1:
                _targets = _targets[0]
            list_targets.append(_targets)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            batch_loss = self.loss(outputs, *list_targets)
            batch_loss.backward()
            self.optimizer.step()

            losses.append(float(batch_loss))
            y_pred.extend(outputs.detach().cpu().numpy())
            y_true.extend(list_targets[0].detach().cpu().numpy())

            aux_losses = (self.model.get_aux_losses() if hasattr(self.model, 'get_aux_losses') else dict())
            aux_losses.update(self.loss.get_aux_losses() if hasattr(self.loss, 'get_aux_losses') else dict())

            for name, aux_loss in aux_losses.items():
                if name not in values:
                    values[name] = 0
                values[name] += float(aux_loss) / nb_batch
            iteration += 1
        loss = np.mean(losses)
        for name, metric in self.metrics.items():
            if name not in values:
                values[name] = 0
            values[name] = float(metric(torch.tensor(y_pred), torch.tensor(y_true)))
        pbar.close()
        return loss, values

    def testing(self, loader: DataLoader, saving_dir=None, exp_name=None, **kwargs):
        """ Evaluate the model.

        Parameters
        ----------
        loader: a pytorch DataLoader
        saving_dir: str path to the saving directory
        exp_name: str, name of the experiments that is used to derive the output file name of testing results.
        Returns
        -------
        y: array-like
            the predicted data.
        X: array-like
            the input data.
        y_true: array-like
            the true data if available.
        loss: float
            the value of the loss function if true data availble.
        values: dict
            the values of the metrics if true data availble.
        """
        y, y_true, X, loss, values = self.test(loader)

        if saving_dir is not None:
            if not os.path.isdir(saving_dir):
                subprocess.check_call(['mkdir', '-p', saving_dir])
                self.logger.info("Directory %s created."%saving_dir)
            with open(os.path.join(saving_dir, (exp_name or 'test')+'.pkl'), 'wb') as f:
                pickle.dump({'y_pred': y, 'y_true': y_true, 'loss': loss, 'metrics': values}, f)

        return y, X, y_true, loss, values

    def test(self, loader):
        """ Evaluate the model on the tests or validation data.

        Parameter
        ---------
        loader: a pytorch Dataset
            the data loader.
        Returns
        -------
        y: array-like
            the predicted data.
        y_true: array-like
            the true data
        X: array_like
            the input data
        loss: float
            the value of the loss function.
        values: dict
            the values of the metrics.
        """

        self.model.eval()
        nb_batch = len(loader)
        pbar = tqdm(total=nb_batch, desc="Mini-Batch")
        loss = 0
        values = {}
        visuals = []

        with torch.no_grad():
            y, y_true, X = [], [], []

            for dataitem in loader:
                pbar.update()
                inputs = dataitem.inputs
                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(self.device)
                list_targets = []
                targets = []
                for item in (dataitem.outputs, dataitem.labels):
                    if item is not None:
                        targets.append(item.to(self.device))
                        y_true.extend(item.cpu().detach().numpy())
                if len(targets) == 1:
                    targets = targets[0]
                elif len(targets) == 0:
                    targets = None
                if targets is not None:
                    list_targets.append(targets)

                outputs = self.model(inputs)

                if len(list_targets) > 0:
                    batch_loss = self.loss(outputs, *list_targets)
                    loss += float(batch_loss) / nb_batch

                y.extend(outputs.cpu().detach().numpy())

                if isinstance(inputs, torch.Tensor):
                    X.extend(inputs.cpu().detach().numpy())

                aux_losses = (self.model.get_aux_losses() if hasattr(self.model, 'get_aux_losses') else dict())
                aux_losses.update(self.loss.get_aux_losses() if hasattr(self.loss, 'get_aux_losses') else dict())
                for name, aux_loss in aux_losses.items():
                    name += " on validation set"
                    if name not in values:
                        values[name] = 0
                    values[name] += aux_loss / nb_batch

                # Now computes the metrics with (y, y_true)
                for name, metric in self.metrics.items():
                    name += " on validation set"
                    values[name] = metric(torch.tensor(y), torch.tensor(y_true))
            pbar.close()

        return y, y_true, X, loss, values