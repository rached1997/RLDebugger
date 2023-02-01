import torch.nn
from torch import nn


def get_model_weights_and_biases(model):
    weights = {}
    biases = {}
    for name, param in model.named_parameters():
        if 'weight' in name:
            weights[name.split('.weight')[0]] = param.data
        if 'bias' in name:
            biases[name.split('.bias')[0]] = param.data

    return weights, biases


def get_loss(original_predictions, model_predictions, loss_fn):
    loss_value = loss_fn(torch.Tensor(original_predictions), torch.Tensor(model_predictions)).mean()
    return loss_value


def get_sample(observations, labels, sample_size):
    return observations[:sample_size], labels[:sample_size]


def is_activation_function(layer):
    activations_functions = [nn.ELU, nn.LeakyReLU, nn.ReLU6, nn.SELU, nn.Tanh, nn.Sigmoid, nn.ReLU]
    for act_layer in activations_functions:
        if isinstance(layer, act_layer):
            return True
    return False
