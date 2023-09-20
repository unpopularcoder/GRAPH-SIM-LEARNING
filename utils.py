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
    return np.mean(precision_ks, axis=0), np.mean(inclusive_final_true_ks, axis=0), np.mean(inclusive_final_pred_ks, axis=0)


def metrics_spearmanr_rho(true, predication):
    assert true.shape == predication.shape
    rho, p_val = stats.spearmanr(true, predication)
    return rho


def metrics_kendall_tau(true, predication):
    assert true.shape == predication.shape
    tau, p_val = stats.kendalltau(true, predication)
    return tau


def metrics_mean_square_error(true, predication):
    assert true.shape == predication.shape
    mse = (np.square(true - predication).mean())
    return mse


def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.set_cols_dtype(['t', 't'])
    t.add_rows([['Parameter', 'Value']] + [[k.replace('_', ' ').capitalize(), args[k]] for k in keys])
    print(t.draw())


def top_k_ids(data, k, inclusive, rm):
    """
    :param data: input
    :param k:
    :param inclusive: whether to be tie inclusive or not.
        For example, the ranking may look like this:
        7 (sim_score=0.99), 5 (sim_score=0.99), 10 (sim_score=0.98), ...
        If tie inclusive, the top 1 results are [7, 9].
        Therefore, the number of returned results may be larger than k.
        In summary,
            len(rtn) == k if not tie inclusive;
            len(rtn) >= k if tie inclusive.
    :param rm: 0
    :return: for a query, the ids of the top k database graph
    ranked by this model.
    """
    sort_id_mat = np.argsort(-data)
    n = sort_id_mat.shape[0]
    if k < 0 or k >= n:
        raise RuntimeError('Invalid k {}'.format(k))
    if not inclusive:
        return sort_id_mat[:k]
    # Tie inclusive.
    dist_sim_mat = data
    while k < n:
        cid = sort_id_mat[k - 1]
        nid = sort_id_mat[k]
        if abs(dist_sim_mat[cid] - dist_sim_mat[nid]) <= rm:
            k += 1
        else:
            break
    return sort_id_mat[:k]


def prec_at_ks(true_r, pred_r, ks, rm=0):
    """
    Ranking-based. prec@ks.
    :param true_r: result object indicating the ground truth.
    :param pred_r: result object indicating the prediction.
    :param ks: k
    :param rm: 0
    :return: precision at ks.
    """
    true_ids = top_k_ids(true_r, ks, inclusive=True, rm=rm)
    pred_ids = top_k_ids(pred_r, ks, inclusive=True, rm=rm)
    ps = min(len(set(true_ids).intersection(set(pred_ids))), ks) / ks
    return ps


def ranking_func(data):
    sort_id_mat = np.argsort(-data)
    n = sort_id_mat.shape[0]
    rank = np.zeros(n)
    for i in range(n):
        finds = np.where(sort_id_mat == i)
   