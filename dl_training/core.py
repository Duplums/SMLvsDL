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

class Base:
    """ Class to perform classification.
    """
    def __init__(self, optimizer_name="Adam", learning_rate=1e-3,
                 loss_name="NLLLoss", metrics=None, use_cuda=False,
                 pretrained=None, freeze_until_layer=None, load_optimizer=True, use_multi_gpu=True,
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
            wether to use GPU or CPU.
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
        super().__init__(
            signals=["before_epoch", "after_epoch", "after_iteration"])
        self.optimizer = kwargs.get("optimizer")
        self.logger = logging.getLogger("dl_training")
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
        if freeze_until_layer is not None:
            freeze_until(self.model, freeze_until_layer)

        if use_multi_gpu and torch.cuda.device_count() > 1:
            self.model = DataParallel(self.model)

        self.model = self.model.to(self.device)

    def training(self, manager, nb_epochs: int, checkpointdir=None,
                 fold_index=None, epoch_index=None,
                 scheduler=None, with_validation=True,
                 nb_epochs_per_saving=1, exp_name=None, standard_optim=True,
                 gpu_time_profiling=False, **kwargs_train):
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
        epoch_index: int, default None
            the iteration where to start the counting from
        scheduler: torch.optim.lr_scheduler, default None
            a scheduler used to reduce the learning rate.
        with_validation: bool, default True
            if set use the validation dataset.
        with_visualization: bool, default False,
            whether it uses a visualizer that will plot the losses/metrics/images in a WebApp framework
            during the training process
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
        if epoch_index is None:
            epoch_index = 0
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
                self.notify_observers("before_epoch", epoch=epoch, fold=fold)
                loss, values = self.train(loader.train, train_visualizer, fold, epoch,
                                          standard_optim=standard_optim,
                                          gpu_time_profiling=gpu_time_profiling, **kwargs_train)

                train_history.log((fold, epoch+epoch_index), loss=loss, **values)
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
                        epoch=epoch+epoch_index,
                        fold=fold,
                        outdir=checkpointdir,
                        name=exp_name,
                        optimizer=self.optimizer)
                    train_history.save(
                        outdir=checkpointdir,
                        epoch=epoch+epoch_index,
                        fold=fold)
                if with_validation:
                    _, _, _, loss, values = self.test(loader.validation,
                                                      standard_optim=standard_optim, **kwargs_train)
                    valid_history.log((fold, epoch+epoch_index), validation_loss=loss, **values)
                    valid_history.summary()
                    if valid_visualizer is not None:
                        valid_visualizer.refresh_current_metrics()
                    if checkpointdir is not None and (epoch % nb_epochs_per_saving == 0 or epoch == nb_epochs-1) \
                            and epoch > 0:
                        valid_history.save(
                            outdir=checkpointdir,
                            epoch=epoch+epoch_index,
                            fold=fold)
                self.notify_observers("after_epoch", epoch=epoch, fold=fold)
        return train_history, valid_history

    def train(self, loader, visualizer=None, fold=None, epoch=None, standard_optim=True,
              gpu_time_profiling=False, **kwargs):
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
        if gpu_time_profiling:
            gpu_time_per_batch = []
        if not standard_optim:
            loss, values = self.model(iter(loader), pbar=pbar, visualizer=visualizer)
        else:
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
                if gpu_time_profiling:
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()

                outputs = self.model(inputs)

                if gpu_time_profiling:
                    end_event.record()
                    torch.cuda.synchronize()
                    elapsed_time_ms = start_event.elapsed_time(end_event)
                    gpu_time_per_batch.append(elapsed_time_ms)

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
                if iteration % 10 == 0:
                    if visualizer is not None:
                        visualizer.refresh_current_metrics()
                        if hasattr(self.model, "get_current_visuals"):
                            visuals = self.model.get_current_visuals()
                            visualizer.display_images(visuals, ncols=3)
                iteration += 1
            loss = np.mean(losses)
            for name, metric in self.metrics.items():
                if name not in values:
                    values[name] = 0
                values[name] = float(metric(torch.tensor(y_pred), torch.tensor(y_true)))

        if gpu_time_profiling:
            self.logger.info("GPU Time Statistics over 1 epoch:\n\t- {:.2f} +/- {:.2f} ms calling model(data) per batch"
                                                              "\n\t- {:.2f} ms total time over 1 epoch ({} batches)".format(
                np.mean(gpu_time_per_batch), np.std(gpu_time_per_batch), np.sum(gpu_time_per_batch), nb_batch))
        pbar.close()
        return loss, values

    def testing(self, loader: DataLoader, with_logit=False, predict=False, with_visuals=False,
                saving_dir=None, exp_name=None, standard_optim=True, **kwargs):
        """ Evaluate the model.

        Parameters
        ----------
        loader: a pytorch DataLoader
        with_logit: bool, default False
            apply a softmax to the result.
        predict: bool, default False
            take the argmax over the channels.
        with_visuals: bool, default False
            returns the visuals got from the model
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
        if with_visuals:
            y, y_true, X, loss, values, visuals = self.test(
                loader, with_logit=with_logit, predict=predict, with_visuals=with_visuals,
                standard_optim=standard_optim)
        else:
            y, y_true, X, loss, values = self.test(
                loader, with_logit=with_logit, predict=predict, with_visuals=with_visuals,
                standard_optim=standard_optim)

        if saving_dir is not None:
            if not os.path.isdir(saving_dir):
                subprocess.check_call(['mkdir', '-p', saving_dir])
                self.logger.info("Directory %s created."%saving_dir)
            with open(os.path.join(saving_dir, (exp_name or 'test')+'.pkl'), 'wb') as f:
                pickle.dump({'y_pred': y, 'y_true': y_true, 'loss': loss, 'metrics': values}, f)
        
        if with_visuals:
            return y, X, y_true, loss, values, visuals

        return y, X, y_true, loss, values

    def test(self, loader, with_logit=False, predict=False, with_visuals=False, standard_optim=True):
        """ Evaluate the model on the test or validation data.

        Parameter
        ---------
        loader: a pytorch Dataset
            the data loader.
        with_logit: bool, default False
            apply a softmax to the result.
        predict: bool, default False
            take the argmax over the channels.

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
            if not standard_optim:
                loss, values, y, y_true, X = self.model(iter(loader), pbar=pbar)
            else:
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
                    if with_visuals:
                        visuals.append(self.model.get_current_visuals())
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
            
            if len(visuals) > 0:
                visuals = np.concatenate(visuals, axis=0)
            try:
                if with_logit:
                    y = func.softmax(torch.tensor(y), dim=1).detach().cpu().numpy()
                if predict:
                    y = np.argmax(y, axis=1)
            except Exception as e:
                print(e)
        if with_visuals:
            return y, y_true, X, loss, values, visuals
        return y, y_true, X, loss, values

    def MC_test(self, loader,  MC=50):
        """ Evaluate the model on the test or validation data by using a Monte-Carlo sampling.

        Parameter
        ---------
        loader: a pytorch Dataset
            the data loader.
        MC: int, default 50
            nb of times to perform a feed-forward per input

        Returns
        -------
        y: array-like dims (n_samples, MC, ...) where ... is the dims of the network's output
            the predicted data.
        y_true: array-like dims (n_samples, MC, ...) where ... is the dims of the network's output
            the true data
        """
        self.model.eval()
        nb_batch = len(loader)
        pbar = tqdm(total=nb_batch, desc="Mini-Batch")

        with torch.no_grad():
            y, y_true = [], []
            for dataitem in loader:
                pbar.update()
                inputs = dataitem.inputs
                if isinstance(inputs, torch.Tensor):
                    inputs = inputs.to(self.device)
                current_y, current_y_true = [], []
                for _ in range(MC):
                    for item in (dataitem.outputs, dataitem.labels):
                        if item is not None:
                            current_y_true.append(item.cpu().detach().numpy())
                    outputs = self.model(inputs)
                    current_y.append(outputs.cpu().detach().numpy())
                y.extend(np.array(current_y).swapaxes(0, 1))
                y_true.extend(np.array(current_y_true).swapaxes(0, 1))
        pbar.close()

        return np.array(y), np.array(y_true)
