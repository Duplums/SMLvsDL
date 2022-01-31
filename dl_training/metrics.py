# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Define common metrics.
"""

# Third party import
import logging
import torch
import numpy as np
import torch.nn.functional as func
import sklearn.metrics as sk_metrics
import scipy, re
from dl_training.utils import get_pickle_obj
import torch.nn as nn
from typing import List, Dict, Sequence
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix
from scipy.special import expit

# Global parameters
logger = logging.getLogger("SMLvsDL")

def get_confusion_matrix(y_pred, y):
    y_pred = y_pred.data.max(dim=1)[1]
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    return confusion_matrix(y, y_pred.detach().cpu().numpy())

def accuracy(y_pred, y):
    if len(y_pred.shape) == 1:
        y_pred = (y_pred > 0)
    y_pred = y_pred.data.max(dim=1)[1]
    return sk_metrics.accuracy_score(y.detach().cpu().numpy(), y_pred.detach().cpu().numpy())

def average_precision_score(y_pred, y):
    return sk_metrics.average_precision_score(y.detach().cpu().numpy(), y_pred.detach().cpu().numpy())


def ECE_score(y_pred, y_true, normalize=False, n_bins=5, verbose=False):
    from sklearn.calibration import calibration_curve
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    frac_pos, mean_pred_proba = calibration_curve(y_true, y_pred, normalize=normalize, n_bins=n_bins)
    confidence_hist, _ = np.histogram(y_pred, bins=n_bins, range=(0,1))

    if len(confidence_hist) > len(frac_pos):
        if verbose:
            print('Not enough data to compute confidence in all bins. NaN returned')
        return np.nan

    score = np.sum(confidence_hist * np.abs(frac_pos - mean_pred_proba) / len(y_pred))

    return score

def AUCE_score(y_mean, y_std, y_true, n_bins=100):
    """Metric to compute calibration for regression model as defined in
       Evaluating Scalable Bayesian Deep Learning Methods for Robust Computer Vision, CVPR 2020
       We assume y ~ N(y_mean, y_std**2)
    """
    y_mean = np.array(y_mean).reshape(1, -1)
    y_std = np.array(y_std).reshape(1, -1)
    y_true = np.array(y_true).reshape(1, -1)
    confidences = np.arange(1.0/n_bins, 1, 1.0/n_bins).reshape(n_bins-1, 1) # shape (n_bins,1)
    # shape (n_bins, n_samples)
    lower_scores = y_mean - scipy.stats.norm.ppf((confidences+1)/2.0) * y_std # shape (n_bins, n_samples)
    upper_scores = y_mean + scipy.stats.norm.ppf((confidences+1)/2.0) * y_std # shape (n_bins, n_samples)
    coverage = np.count_nonzero(np.logical_and(y_true >= lower_scores, y_true <= upper_scores), axis=1)/y_mean.size # shape (n_bins,)
    abs_error = np.abs(coverage - confidences.reshape(n_bins-1))
    score = np.trapz(y=abs_error, x=confidences.reshape(n_bins-1))
    return score

def roc_auc(y_pred, y):
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    if len(y_pred.shape) == 2 and y_pred.shape[1] == 2:
        return roc_auc_score(y, y_pred[:,1].detach().cpu().numpy())
    elif len(y_pred.shape) < 2:
        return roc_auc_score(y, y_pred.detach().cpu().numpy())
    else:
        raise ValueError('Invalid shape for y_pred: {}'.format(y_pred.shape))

def balanced_accuracy(y_pred, y):
    if len(y_pred.shape) == 1:
        y_pred = (y_pred > 0)
    if len(y_pred.shape) == 2:
        y_pred = y_pred.data.max(dim=1)[1] # get the indices of the maximum
    return balanced_accuracy_score(y.detach().cpu().numpy(), y_pred.detach().cpu().numpy())

# True Positive Rate = TP/P (also called Recall)
def sensitivity(y_pred, y, positive_label=1):
    y_pred = y_pred.data.max(dim=1)[1]
    TP = (y_pred.eq(y) & y.eq(positive_label)).sum().cpu().numpy()
    P = y.eq(positive_label).sum().cpu().numpy()
    if P == 0:
        return 0.0
    return float(TP/P)

# True Negative Rate = TN/N
def specificity(y_pred, y, negative_label=0):
    y_pred = y_pred.data.max(dim=1)[1]
    TN = (y_pred.eq(y) & y.eq(negative_label)).sum().cpu().numpy()
    N = y.eq(negative_label).sum().cpu().numpy()
    if N == 0:
        return 0.0
    return float(TN/N)


def RMSE(y_pred, y):
    rmse = torch.sqrt(torch.mean((y_pred - y)**2)).detach().cpu().numpy()
    return float(rmse)


METRICS = {
    "accuracy": accuracy,
    "balanced_accuracy": balanced_accuracy,
    "average_precision_score": average_precision_score,
    "RMSE": RMSE,
    "specificity": specificity,
    "sensitivity": sensitivity,
    "confusion_matrix": get_confusion_matrix,
    "roc_auc": roc_auc,
    # cf. scikit doc: " The binary case expects a shape (n_samples,), and the scores
    # must be the scores of the class with the greater label."
}

def get_multiclass_metrics(path, epochs_tested, folds_tested, normalize=True, display=True):
    test_results = [get_pickle_obj(path.format(fold=f, epoch=e)) for (f, e) in zip(folds_tested, epochs_tested)]
    try:
        test_results = [{'y_pred': expit(np.array(t['y_pred'])) if normalize else np.array(t['y_pred']),
                         'y_true': np.array(t['y_true'])} for t in test_results]
    except KeyError:
        test_results = [{'y_pred': np.array(t['y']), 'y_true': np.array(t['y_true'])} for t in test_results]

    b_acc = [balanced_accuracy_score(t['y_true'], np.argmax(np.array(t['y_pred']), axis=1)) for t in test_results]
    acc = [sk_metrics.accuracy_score(t['y_true'], np.argmax(np.array(t['y_pred']), axis=1)) for t in test_results]
    if display:
        print("BAcc = {} +/- {}\nAcc = {} +/- {}".format(np.mean(b_acc), np.std(b_acc), np.mean(acc),np.std(acc)))
    return dict(bacc=b_acc, acc=acc, confusion_matrix=[confusion_matrix(t['y_true'],
                                                                        np.argmax(np.array(t['y_pred']), axis=1))
                                                       for t in test_results])

def get_binary_classification_metrics(path: str, epochs_tested: Sequence[int], folds_tested: Sequence[int],
                                      MCTest: bool=False, Ensembling: bool=False,
                                      normalize: bool=True, y_pred_key: str=None,
                                      softmax_applied=False, display: bool=True,
                                      mask=None) -> Dict[str, List[float]]:
    """
    :param path: file path with convention r".*{fold}.*{epoch}.*"
    :param epochs_tested: list of epochs to tests
    :param folds_tested: list of matching folds to tests (same len as epoch)
    :param MCTest: whether we performed pynet <MCTest> (several forward passes for same x)
    :param Ensembling: whether we performed ensembling tests
    :param normalize: if True, apply sigmoid function to predictions
    :param y_pred_key: name of the key to look for in the loading dict
    :param softmax_applied: if True, we assume each prediction is formatted as [p(y=0), p(y=1)] (results of softmax).
    It is mutually exclusive with normalize
    :param display: if True, print the metrics
    :param mask: mask to apply to the tested samples
    :return: dict containing the metrics
    """
    import sklearn
    assert len(epochs_tested) == len(folds_tested), "Invalid nb of epochs or folds"
    assert re.match(r".*{fold}.*{epoch}.*", path), "Unknown path format: %s"%path
    assert softmax_applied + normalize < 2, "<normalize>, <softmax_applied> mutually exclusive"

    test_results = [get_pickle_obj(path.format(fold=f, epoch=e)) for (f, e) in zip(folds_tested, epochs_tested)]
    entropy_func = lambda sigma: - ((1 - sigma) * np.log(1 - sigma + 1e-8) + sigma * np.log(sigma + 1e-8))

    if y_pred_key is None:
        if MCTest or Ensembling:
            y_pred_key = "y"
        else:
            y_pred_key = "y_pred"
    ## MC Tests
    if MCTest or Ensembling:
        MI = np.concatenate(
            [entropy_func(expit(np.array(t[y_pred_key])) if normalize else np.array(t[y_pred_key]).mean(axis=1)) -
             entropy_func(expit(np.array(t[y_pred_key])) if normalize else np.array(t[y_pred_key])).mean(axis=1)
             for i, t in enumerate(test_results)])
        test_results = [{'y_pred': (expit(np.array(t[y_pred_key])) if normalize else np.array(t[y_pred_key])).mean(axis=1),
                         'y_true': np.array(t['y_true'])[:, 0]} for t in
                        test_results]
        assert all(np.array_equal(np.array(t['y_true']), np.array(test_results[0]['y_true'])) for t in test_results)
        print('MI = {} +/- {}'.format(np.mean(MI), np.std(MI)))

    else:
        test_results = [{'y_pred': expit(np.array(t[y_pred_key])) if normalize else np.array(t[y_pred_key]),
                         'y_true': np.array(t['y_true'])} for t in test_results]
    if softmax_applied:
        for i, t in enumerate(test_results):
            test_results[i]["y_pred"] = t["y_pred"][:, 1]
    if mask is not None:
        test_results = [{"y_pred": t["y_pred"][mask], "y_true": t["y_true"][mask]} for t in test_results]
    y_pred, y_true = np.array([t['y_pred'] for t in test_results]), np.array([t['y_true'] for t in test_results])
    all_ece = [ECE_score(t['y_pred'], t['y_true'], verbose=display) for t in test_results]
    #H_pred = entropy_func(y_pred)
    mask_corr = [(pred > 0.5) == true for (pred, true) in zip(y_pred, y_true)]
    #H_pred_corr = np.concatenate([H_pred[f][mask_corr[f]] for f in range(len(folds_tested))])
    #H_pred_incorr = np.concatenate([H_pred[f][~mask_corr[f]] for f in range(len(folds_tested))])
    all_auc = [roc_auc_score(t['y_true'], np.array(t['y_pred'])) for t in test_results]
    all_aupr_success = [sklearn.metrics.average_precision_score(t['y_true'], t['y_pred']) for t in test_results]
    all_aupr_error = [sklearn.metrics.average_precision_score(1-np.array(t['y_true']), -np.array(t['y_pred'])) for t in test_results]

    all_confusion_matrix = [confusion_matrix(t['y_true'], np.array(t['y_pred']) > 0.5) for t in test_results]
    recall = [[m[i, i] / m[i, :].sum() for m in all_confusion_matrix] for i in range(2)]
    precision = [m[1, 1] / m[:, 1].sum() for m in all_confusion_matrix]
    bacc = [balanced_accuracy_score(t['y_true'], np.array(t['y_pred']) > 0.5) for t in test_results]
    acc = [sk_metrics.accuracy_score(t['y_true'], np.array(t['y_pred']) > 0.5) for t in test_results]
    if display:
        print('Mean AUC= {} +/- {} \nMean Balanced Acc = {} +/- {}\nMean Acc = {} +/- {}\n'
              'Mean Recall_+ = {} +/- {}\nMean Recall_- = {} +/- {}\n'
              'Mean Precision = {} +/- {}\nMean AUPR_Success = {} +/- {}\n'
              'Mean AUPR_Error = {} +/- {}\nMean ECE = {} +/- {}'.
              format(np.mean(all_auc), np.std(all_auc), np.mean(bacc), np.std(bacc), np.mean(acc), np.std(acc),
                     np.mean(recall[1]), np.std(recall[1]),
                     np.mean(recall[0]), np.std(recall[0]), np.mean(precision), np.std(precision),
                     #H_pred_corr.mean(), H_pred_corr.std(), H_pred_incorr.mean(), H_pred_incorr.std(),
                     np.mean(all_aupr_success), np.std(all_aupr_success), np.mean(all_aupr_error), np.std(all_aupr_error),
                     np.mean(all_ece), np.std(all_ece)))
    return dict(auc=all_auc, bacc=bacc, recall_pos=recall[1], recall_neg=recall[0], precision=precision,
                aupr_success=all_aupr_success, aupr_error=all_aupr_error, ece=all_ece)


def get_regression_metrics(path, epochs_tested, folds_tested, display=True, y_pred_key="y_pred", mask=None):
    from sklearn.linear_model import LinearRegression
    from scipy.stats import pearsonr

    data =  [get_pickle_obj(path.format(fold=f, epoch=e)) for (f, e) in zip(folds_tested, epochs_tested)]
    data = [(np.array(d['y_true']).reshape(-1, 1), np.array(d[y_pred_key]).reshape(-1, 1)) for d in data]
    if mask is not None:
        data = [(y_true[mask], y_pred[mask]) for (y_true, y_pred) in data]

    regs = [LinearRegression().fit(y_true, y_pred) for (y_true, y_pred) in data]
    correlations = [pearsonr(y_pred.flatten(), y_true.flatten())[0] for (y_true, y_pred) in data]
    MAEs = [np.mean(np.abs(y_pred - y_true)) for (y_true, y_pred) in data]
    RMSEs = [np.sqrt(np.mean(np.abs(y_pred - y_true) ** 2)) for (y_true, y_pred) in data]
    R2s = [reg.score(y_true, y_pred) for (reg, (y_true, y_pred)) in zip(regs, data)]
    if display:
        print('MAE = {} +/- {}\nRMSE = {} +/- {}\nR^2 = {} +/- {}\nr = {} +/- {}'.format(np.mean(MAEs), np.std(MAEs),
                                                                                         np.mean(RMSEs), np.std(RMSEs),
                                                                                         np.mean(R2s), np.std(R2s),
                                                                                         np.mean(correlations), np.std(correlations)))
    return dict(mae=MAEs, rmse=RMSEs, R2=R2s, r=correlations)