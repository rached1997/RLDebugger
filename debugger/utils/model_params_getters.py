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
    loss_value = loss_fn(original_predictions, model_predictions).mean()
    return loss_value


def get_sample(observations, labels, sample_size):
    return observations[:sample_size], labels[:sample_size]


def is_activation_function(layer):
    activations_functions = [nn.ELU, nn.LeakyReLU, nn.ReLU6, nn.SELU, nn.Tanh, nn.Sigmoid, nn.ReLU]
    for act_layer in activations_functions:
        if isinstance(layer, act_layer):
            return True
    return False


def get_last_layer_activation(model):
    # Get the last layer of the model
    layers = list(model.children())
    last_layer = layers[-1]

    # Check if the last layer has an activation function
    if isinstance(last_layer, nn.Module) and hasattr(last_layer, 'activation'):
        # Get the name of the activation function
        activation_name = last_layer.activation.__class__.__name__
    else:
        activation_name = 'None'
    return activation_name
