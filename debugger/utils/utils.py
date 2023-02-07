from itertools import groupby
from operator import itemgetter

import numpy as np
from scipy.stats import mannwhitneyu
import torch


def almost_equal(value1, value2, rtol=1e-2):
    """
    This function checks if two values are almost equal within a relative tolerance.

    Args:
        value1: first value
        value2: second value
        rtol (float): relative tolerance for the comparison (default: 1e-2)

    Returns:
        True if the values are almost equal, False otherwise
    """
    rerr = torch.abs(value1 - value2)
    if isinstance(value1, torch.Tensor):
        return (rerr <= torch.tensor(rtol)).all()
    else:
        return rerr <= rtol


def is_non_2d(array):
    """
    This function checks if an array is not a 2D array.

    Args:
        array: the input array

    Returns:
        True if the array is not 2D, False otherwise
    """
    return len(array.shape) > 2


def smoothness(data):
    """
    This function calculates the smoothness of a 1D array of values.

    Args:
        data: the 1D array

    Returns:
        the smoothness of the input data
    """
    data_size = len(data)
    if data_size < 1:
        return 1.0
    ratios = (data[1:] / data[:-1])
    rate_changes = np.abs(np.diff(ratios > 1.))
    rate_changes_count = np.count_nonzero(rate_changes)
    return (data_size - rate_changes_count) / data_size


def pure_f_test(data, ref_std, alpha=0.1):
    """
    This function performs the F-test to determine if the variance of two data sets is significantly different.

    Args:
        data: the data to be tested
        ref_std: the standard deviation of the second data set
        alpha: the significance level for the test (default: 0.1)

    Returns:
        a tuple of the calculated F-value and a boolean indicating if the variances are significantly different
    """
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

    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    var_1 = np.std(data)**2
    var_2 = ref_std ** 2
    F = var_1 / var_2 if var_1 > var_2 else var_2 / var_1
    return F, F <= _F_critical(alpha)


def are_significantly_different(sample_1, sample_2, alpha=0.05):
    """
    This function performs a Mann-Whitney U test to determine if two samples are significantly different.

    Args:
        sample_1: first sample
        sample_2: second sample
        alpha: the significance level for the test (default: 0.05)

    Returns:
        True if the samples are significantly different, False otherwise
    """
    stat, p = mannwhitneyu(sample_1, sample_2)
    return p <= alpha


def get_activation_max_min_bound(name):
    """
    This function returns the minimum and maximum bounds of an activation function.

    Args:
        name: the name of the activation function

    Returns:
        a tuple of the minimum and maximum bounds of the activation function
    """
    name = name[:name.rfind('_')]
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
    """
    This function calculates the probabilities of the different targets.

    Args:
        targets: the targets array

    Returns:
        the probabilities of the different targets
    """
    if targets.shape[1] == 1:
        labels_probas = np.zeros(2)
        labels_probas[0] = torch.mean(1.0 - targets)
        labels_probas[1] = torch.mean(targets)
    else:
        labels_probas = torch.mean(targets, dim=0)
    return labels_probas


def get_balance(targets):
    """
    This function calculates the balance of the targets.

    Args:
        targets: the targets array

    Returns:
        the balance of the targets
    """
    if targets.shape[1] == 1:
        labels = 2
    else:
        labels = targets.shape[1]
    labels_probas = get_probas(targets)
    perplexity = torch.exp(torch.distributions.Categorical(labels_probas, validate_args=False).entropy())
    balance = (perplexity - 1) / (labels - 1)
    return balance


def compute_ro_B(activations, min_out, max_out, bins_count):
    """
    This function calculates the value of Ro_B for a set of activations.

    Args:
        activations: the activations array
        min_out: the minimum activation value
        max_out: the maximum activation value
        bins_count: the number of bins to divide the activation values into

    Returns:
        The values of Ro_b
    """
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
    """
    Reshape a 2D numpy array into a 2D matrix with the specified dimension kept.

    Args:
        array: The 2D numpy array to be reshaped.
        keep: Specifies which dimension to keep after reshaping. Must be either 'first' or 'last'. Defaults to 'first'.

    Returns:
        The reshaped 2D numpy array.
    """
    if keep == 'first':
        return array.reshape(array.shape[0], -1)
    elif keep == 'last':
        return array.reshape(-1, array.shape[-1])


