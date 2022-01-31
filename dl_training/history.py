# -*- coding: utf-8 -*-
"""
Module that provides logging utilities.
"""


# System import
import os
import logging
import collections
import time
import datetime
import pickle

# Third party import
import numpy as np
from tabulate import tabulate


# Global parameters
logger = logging.getLogger("SMLvsDL")


class History(object):
    """ Track training progress by following the some metrics.
    """
    def __init__(self, name, verbose=0):
        """ Initilize the class.

        Parameters
        ----------
        name: str
            the object name.
        verbose: int, default 0
            control the verbosity level.
        """
        self.name = name
        self.verbose = verbose
        self.step = None
        self.metrics = set()
        self.history = collections.OrderedDict()

    def __repr__(self):
        """ Display the history.
        """
        table = []
        for step in self.steps:
            values = []
            for metric in self.metrics:
                values.append(self.history[step][metric])
            table.append([step] + values)
        return tabulate(table, headers=self.metrics)


    def log(self, step, **kwargs):
        """ Record some metrics at a specific step.

        Example:
            state = History()
            state.log(1, loss=1., accuracy=0.)

        If logging the same metrics for one specific step, new values
        overwrite older ones.

        Parameters
        ----------
        step: int or uplet
            The step name: we can use a tuple to log the fold, the epoch
            or the step within the epoch.
        kwargs
            The metrics to be logged.
        """
        if not isinstance(step, (int, tuple)):
            raise ValueError("Step must be an int or a tuple.")
        self.step = step
        self.metrics |= set(kwargs.keys())
        if step not in self.history:
            self.history[step] = {}
        for key, val in kwargs.items():
            self.history[step][key] = val
        self.history[step]["__timestamp__"] = time.time()

    @property
    def steps(self):
        """ Returns a list of all steps.
        """
        return list(self.history.keys())

    def __getitem__(self, metric, with_nan=False):

        steps = []
        data = []
        for step in self.steps:
            if metric in self.history[step] or with_nan:
                steps.append(step)
                data.append(self.history[step].get(metric))
        return steps, data

    def summary(self):
        last_step = self.steps[-1]
        msg = "{:6s} {:15s}".format(self.name, repr(last_step))
        for key in self.metrics:
            if key in self.history[last_step]:
                msg += "{:6s}:{:10f}  ".format(key, self.history[last_step][key])
        msg += "{}".format(str(self.get_total_time()))
        logger.info(msg)

    def get_total_time(self):
        """ Returns the total period between the first and last steps.
        """
        seconds = (
            self.history[self.steps[-1]]["__timestamp__"]
            - self.history[self.steps[0]]["__timestamp__"])
        return datetime.timedelta(seconds=seconds)

    def save(self, outdir, fold, epoch):
        outfile = os.path.join(
            outdir, "{0}_{1}_epoch_{2}.pkl".format(self.name, fold, epoch))
        with open(outfile, "wb") as open_file:
            pickle.dump(self, open_file)

    @classmethod
    def load(cls, file_name, folds=None):
        # folds: (int) if given, load all the files corresponding to the given folds and merge them
        if folds is None:
            with open(file_name, "rb") as open_file:
                return pickle.load(open_file)
        else:
            histories = []
            for k in np.sort(folds):
                with open(file_name%k, 'rb') as open_file:
                    histories.append(pickle.load(open_file))
            return cls.merge_histories(histories, folds=np.sort(folds))

    @classmethod
    def merge_histories(cls, histories, folds=None):
        if len(histories) == 0: return None
        merged = cls(histories[0].name, verbose=histories[0].verbose)
        for k, h in enumerate(histories):
            for step in h.steps:
                if folds is not None:
                    if type(step) == tuple and step[0] != folds[k]:
                        continue
                merged.log(step, **h.history[step])
        return merged

    @classmethod
    def load_from_dir(cls, outdir, name, fold, epoch):
        return cls.load(os.path.join(outdir, "{0}_{1}_epoch_{2}.pkl".format(name, fold, epoch)))

    def get_best_epochs(self, metric, highest=True):
        # Returns a list of n epochs (where n==nb of folds) where each epoch is the best for a given fold according
        # to a metric. If 'highest' is True, the highest score is the best.
        # for each fold, get the best epoch according to the selected metric
        M = self.to_dict(patterns_to_del=['validation_', ' on validation set'])
        assert metric in list(M.keys()), "Unknown metric %s"%metric
        best_epochs = np.argmax(M[metric], axis=1) if highest else np.argmin(M[metric], axis=1)
        return best_epochs

    def to_dict(self, patterns_to_del=None, drop_last=False):
        import re
        # Returns a dictionary {k: M} where k is a metric and M is a matrix n x p where n==nb of folds, p==nb of epochs
        # If one fold is incomplete and drop_last==True, drop it. Otherwise, it won't be matrices but lists of lists.
        # Optionally, <patterns_to_del> can be a list of regex pattern to delete from the metrics.

        if patterns_to_del is not None:
            if isinstance(patterns_to_del, str):
                pattern = re.compile(patterns_to_del)
            elif isinstance(patterns_to_del, list):
                pattern = re.compile('({})'.format('|'.join(patterns_to_del)))
        this_dict = dict()
        # Constructs the dictionary
        for step, metrics in self.history.items():
            for (metric, val) in metrics.items():
                if metric in self.metrics:
                    if patterns_to_del is not None:
                        metric = pattern.sub('', metric)
                    if metric not in this_dict:
                        this_dict[metric] = []
                    if isinstance(step, int):
                        this_dict[metric].append(val)
                    elif isinstance(step, tuple):
                        fold = step[0]
                        if len(this_dict[metric]) <= fold:
                            this_dict[metric].extend([[] for _ in range(fold-len(this_dict[metric])+1)])
                        this_dict[metric][fold].append(val)
        # Checks the structure
        length_per_fold = {m: np.array([len(f) for f in this_dict[m]]) for m in this_dict.keys()}
        if drop_last:
            for m in length_per_fold:
                if len(length_per_fold[m]) > 0:
                    assert np.all(length_per_fold[m][:-1] == length_per_fold[m][0])
                    if not np.all(length_per_fold[m] == length_per_fold[m][0]):
                        del this_dict[m][-1]
        return this_dict