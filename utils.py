"""
Part of the code in <utils.py> is from SimGNN@benedekrozemberczki
"""

import os
import numpy as np
from texttable import Texttable
from scipy import stats


def computing_precision_ks(trues, predictions, ks, inclusive=True, rm=0):
    assert trues.shape == predictions.shape
    m, n = trues.shape

    precision_ks = np.zeros((m, len(ks)))
    inclusive_final_true_ks = np.zeros((m, len(ks)))
    inclusive_final_pred_ks = np.zeros((m, len(ks)))

    for i in range(m):

        for k_idx, k in enumerate(ks):
            assert (type(k) is int and 0 < k < n)
            true_ids, true_k = top_k_ids(trues, i, k, inclusive, rm)
            pred_ids, pred_k = top_k_ids(predictions, i, k, inclusive, rm)
            precision_ks[i][k_idx] = min(len(set(true_ids).intersection(set(pred_ids))), k) / k
            inclusive_final_true_ks[i][k_idx] = true_k
            inclusive_final_pred_ks[i][k_idx] = pred_k
    return n