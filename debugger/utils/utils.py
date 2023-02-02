from itertools import groupby
from operator import itemgetter

import torch.nn as nn
import re
import numpy as np
from scipy.stats import mannwhitneyu
import torch


def almost_equal(value1, value2, rtol=1e-2):
    rerr = torch.abs(value1 - value2)
    if isinstance(value1, torch.Tensor):
        return (rerr <= torch.tensor(rtol)).all()
    else:
        return rerr <= rtol


def is_non_2d(array):
    return len(array.shape) > 2


def smoothness(data):
    data_size = len(data)
    if data_size < 1:
        return 1.0
    ratios = (data[1:] / data[:-1])
    rate_changes = np.abs(np.diff(ratios > 1.))
    rate_changes_count = np.count_nonzero(rate_changes)
    return (data_size - rate_changes_count) / data_size


def pure_f_test(data, ref_std, alpha=0.1):
    def _F_critical(alpha):
        # http://socr.ucla.edu/Applets.dir/F_Table.html
        if alpha == 0.1:
            return 2.70554
        elif alpha == 0.05:
            return 3.8415
        elif alpha == 0.025:
            return 5.0239
        elif alpha == 0.01:
            return 6.635

    var_1 = torch.var(data)
    var_2 = ref_std ** 2
    F = var_1 / var_2 if var_1 > var_2 else var_2 / var_1
    return F, F <= _F_critical(alpha)


def are_significantly_different(sample_1, sample_2, alpha=0.05):
    stat, p = mannwhitneyu(sample_1, sample_2)
    return p <= alpha


def get_activation_max_min_bound(name):
    name = re.sub(r'\([^()]*\)', '', name)
    if name == 'ELU':
        activation_max_bound = +np.inf
        activation_min_bound = -1.0
    elif name == 'LeakyReLU':
        activation_max_bound = +np.inf
        activation_min_bound = -np.inf
    elif name == 'ReLU6':
        activation_max_bound = 6.0
        activation_min_bound = 0.0
    elif name == 'SELU':
        activation_max_bound = +np.inf
        activation_min_bound = -np.inf
    elif name == 'Tanh':
        activation_max_bound = 1.0
        activation_min_bound = -1.0
    elif name == 'Sigmoid':
        activation_max_bound = 1.0
        activation_min_bound = 0.0
    elif name == 'ReLU':
        activation_max_bound = +np.inf
        activation_min_bound = 0.0
    else:
        activation_max_bound = +np.inf
        activation_min_bound = -np.inf
    return activation_max_bound, activation_min_bound


def get_probas(targets):
    if targets.shape[1] == 1:
        labels_probas = np.zeros(2)
        labels_probas[0] = torch.mean(1.0 - targets)
        labels_probas[1] = torch.mean(targets)
    else:
        labels_probas = torch.mean(targets, dim=0)
    return labels_probas


def get_balance(targets):
    if targets.shape[1] == 1:
        labels = 2
    else:
        labels = targets.shape[1]
    labels_probas = get_probas(targets)
    perplexity = torch.exp(torch.distributions.Categorical(labels_probas, validate_args=False).entropy())
    balance = (perplexity - 1) / (labels - 1)
    return balance


def compute_ro_B(activations, min_out, max_out, bins_count):
    bin_size = (max_out - min_out) / bins_count
    bins = np.arange(min_out, max_out, bin_size).tolist()
    divided_values = np.digitize(activations, bins)
    data = [(neu_act, bin_v) for neu_act, bin_v in zip(divided_values, activations)]
    data = list(zip(divided_values, activations))
    grouped_data = [list(map(lambda x: x[1], group)) for _, group in groupby(sorted(data), key=itemgetter(0))]
    f_g = [(len(values), np.mean(values)) for values in grouped_data]
    f_g_prime = np.array([(f_b, np.abs(2 * (g_b - min_out) / (max_out - min_out) - 1) * f_b) for f_b, g_b in f_g])
    return f_g_prime[:, 1].sum() / f_g_prime[:, 0].sum()


def transform_2d(array, keep='first'):
    if keep == 'first':
        return array.reshape(array.shape[0], -1)
    elif keep == 'last':
        return array.reshape(-1, array.shape[-1])


