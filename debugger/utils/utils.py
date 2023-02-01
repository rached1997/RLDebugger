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
