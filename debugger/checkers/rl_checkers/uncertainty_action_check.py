import numpy as np

from debugger.debugger_interface import DebuggerInterface
import torch
import random
from torch import nn
import torch.nn.functional as F


class Memory:
    def __init__(self, max_size):
        self.buffer = [None] * max_size
        self.max_size = max_size
        self.index = 0
        self.size = 0

    def append(self, obj):
        self.buffer[self.index] = obj
        self.size = min(self.size + 1, self.max_size)
        self.index = (self.index + 1) % self.max_size

    def append_batch(self, objs):
        for obj in objs:
            self.append(obj)

    def sample(self, batch_size):
        indices = random.sample(range(self.size), batch_size)
        result = np.stack([self.buffer[index] for index in indices], axis=0).squeeze()
        return torch.tensor(result, device='cuda')


class DropoutWrapper(nn.Module):
    def __init__(self, module, last_layer_name, p=0.5):
        super(DropoutWrapper, self).__init__()
        self.module = module
        self.dropout_rate = p
        self.last_layer_name = last_layer_name

    def forward(self, x):
        for name, layer in self.module.named_modules():
            if len(list(layer.children())) > 0:
                continue
            if isinstance(layer, nn.Dropout):
                continue
            x = layer(x)
            if (name != self.last_layer_name) and (isinstance(layer, nn.Linear)):
                x = F.dropout(x, p=self.dropout_rate, training=True)
        return x


def get_config():
    """
        Return the configuration dictionary needed to run the checkers.

        Returns:
            config (dict): The configuration dictionary containing the necessary parameters for running the checkers.
    """
    config = {
        "Period": 1000,
        "start": 1000,
        "num_repetitions": 100,
        "std_threshold": 0.5,
        "buffer_max_size": 1000,
        "batch_size": 32
    }
    return config


class UncertaintyActionCheck(DebuggerInterface):
    """
    #
    """
    def __init__(self):
        super().__init__(check_type="UncertaintyAction", config=get_config())
        self._buffer = Memory(max_size=self.config["buffer_max_size"])

    def run(self, model, observations, environment):
        """
        #
        """
        observation_shape = environment.observation_space.shape
        if observations.shape == observation_shape:
            self._buffer.append(observations)
        else:
            self._buffer.append_batch(observations)
        if self.check_period() and self.iter_num >= self.config["start"]:
            last_layer_name, _ = list(model.named_modules())[-1]
            observations_batch = self._buffer.sample(batch_size=self.config["batch_size"])
            self.check_mont_carlo_dropout_uncertainty(model, observations_batch, last_layer_name)

    def check_mont_carlo_dropout_uncertainty(self, model, observations, last_layer_name):
        predictions = []
        mcd_model = DropoutWrapper(model, last_layer_name)
        # perform Monte Carlo dropout on the wrapped model
        for i in range(self.config['num_repetitions']):
            output = mcd_model(observations)
            predictions.append(output)

        # Average the predictions to get the final prediction and uncertainty estimate
        uncertainty = torch.mean(torch.std(torch.stack(predictions), dim=0), dim=0)
        if torch.any(uncertainty > self.config["std_threshold"]):
            self.error_msg.append(self.main_msgs['mcd_uncertainty'].format(torch.mean(uncertainty),
                                                                           self.config["std_threshold"],
                                                                           self.config['num_repetitions']))
        return None
