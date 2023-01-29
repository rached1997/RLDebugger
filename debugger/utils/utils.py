# TODO: remove unused functions
import scipy
import numpy as np
import torch.nn as nn
import re
import torch


def add_extra_feeds(feeds, extra_feed_dict):
    if extra_feed_dict == {}:
        return feeds
    d = {}
    d.update(feeds)
    d.update(extra_feed_dict)
    return d


def almost_equal(value1, value2, rtol=1e-2):
    rerr = np.abs(value1 - value2)
    if isinstance(value1, np.ndarray):
        return (rerr <= rtol).all()
    else:
        return rerr <= rtol


def reduce_data(np_data, reductions, axis=None):
    data_reductions = {}
    for reduction_name in reductions:
        data_reductions[reduction_name] = getattr(np, reduction_name)(np_data, axis=axis)
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


def numpify(t):
    """Convert object to a numpy array.

    Args:
        t (np.ndarray | torch.Tensor | obj): Converts object to :py:class:`np.ndarray`.
    """
    if isinstance(t, np.ndarray):
        return t
    elif isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    else:
        return np.array(t)
