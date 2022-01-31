# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2019 - 2020
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

import logging
import torch
import torch.nn as nn
import torch.nn.functional as func
from sklearn.metrics.pairwise import rbf_kernel, cosine_similarity, polynomial_kernel, linear_kernel, \
    manhattan_distances, laplacian_kernel
from sklearn.metrics import pairwise_distances
import numpy as np

# Global parameters
logger = logging.getLogger("SMLvsDL")

class WeaklySupervisedNTXenLoss(nn.Module):
    """
    This loss is proposed in B. Dufumier, Contrastive Learning with Continuous Proxy Meta-Data for 3D MRI Classification, MICCAI 2021
    """
    def __init__(self, kernel='rbf', temperature=0.1, return_logits=False, sigma=1.0):
        """
        :param kernel: a callable function f: [K, *] x [K, *] -> [K, K]
                                              y1, y2          -> f(y1, y2)
                        where (*) is the dimension of the labels (yi)
        default: an rbf kernel parametrized by 'sigma' which corresponds to gamma=1/(2*sigma**2)

        :param temperature:
        :param return_logits:
        """

        # sigma = prior over the label's range
        super().__init__()
        self.kernel = kernel
        self.sigma = sigma
        if self.kernel == 'rbf':
            self.kernel = lambda y1, y2: rbf_kernel(y1, y2, gamma=1./(2*self.sigma**2))
        elif self.kernel == "discrete":
            self.kernel = WeaklySupervisedNTXenLoss.discrete_kernel
        if self.kernel == 'cosine':
            self.kernel = lambda y1, y2: cosine_similarity(y1, y2)
        elif self.kernel == 'linear':
            self.kernel = lambda y1, y2: linear_kernel(y1, y2)
        elif self.kernel == 'polynomial':
            self.kernel = lambda y1, y2: polynomial_kernel(y1, y2, gamma=1/self.sigma)
        elif self.kernel == "manhattan":
            self.kernel = lambda y1, y2: manhattan_distances(y1, y2)
        elif self.kernel == 'laplacian':
            self.kernel = lambda y1, y2: laplacian_kernel(y1, y2, gamma=1/self.sigma)
        else:
            assert hasattr(self.kernel, '__call__'), 'kernel must be a callable'
        self.temperature = temperature
        self.return_logits = return_logits
        self.INF = 1e8

    @staticmethod
    def discrete_kernel(y1, y2):
        """
        :param y1: matrix shape [N, *]
        :param y2: matrix shape [N, *]
        :return: matrix M shape [N, N] where M[i][j] = 1({y1[i] == y2[j]})
        """
        M = (pairwise_distances(y1, y2, metric="hamming") == 0) # dist = proportion of components disageeing
        return M.astype(np.float)

    def forward(self, z_i, z_j, labels):
        N = len(z_i)
        assert N == len(labels), "Unexpected labels length: %i"%len(labels)
        z_i = func.normalize(z_i, p=2, dim=-1) # dim [N, D]
        z_j = func.normalize(z_j, p=2, dim=-1) # dim [N, D]
        sim_zii= (z_i @ z_i.T) / self.temperature # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zjj = (z_j @ z_j.T) / self.temperature # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zij = (z_i @ z_j.T) / self.temperature # dim [N, N] => the diag contains the correct pairs (i,j) (x transforms via T_i and T_j)
        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_zii = sim_zii - self.INF * torch.eye(N, device=z_i.device)
        sim_zjj = sim_zjj - self.INF * torch.eye(N, device=z_i.device)

        all_labels = labels.view(N, -1).repeat(2, 1).detach().cpu().numpy() # [2N, *]
        weights = self.kernel(all_labels, all_labels) # [2N, 2N]
        weights = weights * (1 - np.eye(2*N)) # puts 0 on the diagonal
        weights /= weights.sum(axis=1)
        # if 'rbf' kernel and sigma->0, we retrieve the classical NTXenLoss (without labels)
        sim_Z = torch.cat([torch.cat([sim_zii, sim_zij], dim=1),
                           torch.cat([sim_zij.T, sim_zjj], dim=1)], dim=0) # [2N, 2N]
        log_sim_Z = func.log_softmax(sim_Z, dim=1)

        loss = -1./N * (torch.from_numpy(weights).to(z_i.device) * log_sim_Z).sum()

        correct_pairs = torch.arange(N, device=z_i.device).long()

        if self.return_logits:
            return loss, sim_zij, correct_pairs

        return loss

    def __str__(self):
        return "{}(temp={}, kernel={}, sigma={})".format(type(self).__name__, self.temperature,
                                                         self.kernel.__name__, self.sigma)
