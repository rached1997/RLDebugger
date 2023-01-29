# TODO: remove unused functions
import scipy
import numpy as np
import torch.nn as nn
import re
import torch
import scipy.stats as stats


def add_extra_feeds(feeds, extra_feed_dict):
    if extra_feed_dict == {}:
        return feeds
    d = {}
    d.update(feeds)
    d.update(extra_feed_dict)
    return d


def almost_equal(value1, value2, rtol=1e-2):
    rerr = torch.abs(value1 - value2)
    if isinstance(value1, torch.Tensor):
        return (rerr <= torch.tensor(rtol)).all()
    else:
        return rerr <= rtol


def reduce_data(data, reductions, axis=None):
    data_reductions = {}
    for reduction_name in reductions:
        data_reductions[reduction_name] = getattr(torch, reduction_name)(data, dim=axis)
    return data_reductions


def trim_data(data, trim_prop=0.1):
    data.sort()
    trimmed_data = scipy.stats.trimboth(data.flatten(), trim_prop)
    return trimmed_data


def transform_2d(array, keep='first'):
    if keep == 'first':
        return array.reshape(array.shape[0], -1)
    elif keep == 'last':
        return array.reshape(-1, array.shape[-1])


def is_non_2d(array):
    return len(array.shape) > 2


def readable(float_num):
    return round(float_num, 3)


def is_activation_function(layer):
    activations_functions = [nn.ELU, nn.LeakyReLU, nn.ReLU6, nn.SELU, nn.Tanh, nn.Sigmoid, nn.ReLU]
    for act_layer in activations_functions:
        if isinstance(layer, act_layer):
            return True
    return False


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

