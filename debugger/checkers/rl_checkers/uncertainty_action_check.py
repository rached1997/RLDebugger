import numpy as np

from debugger.config_data_classes.rl_checkers.uncertainty_action_config import UncertaintyActionConfig
from debugger.utils.utils import get_device
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
        result = torch.stack([self.buffer[index] for index in indices], dim=0).squeeze()
        return torch.tensor(result, device=get_device())


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


class UncertaintyActionCheck(DebuggerInterface):
    """
    #
    """

    def __init__(self):
        super().__init__(check_type="UncertaintyAction", config=UncertaintyActionConfig)
        self._buffer = Memory(max_size=self.config.buffer_max_size)

    def run(self, model, observations, environment):
        """
        Checks whether the actions predictions uncertainty over time is reducing during the learning process

        Args:
            model (nn.Module): the main model
            observations (Tensor): the observation collected after doing one-step
            environment (gym.env): the training RL environment
        """
        if self.skip_run(self.config.skip_run_threshold):
            return
        observation_shape = environment.observation_space.shape
        if observations.shape == observation_shape:
            self._buffer.append(observations)
        else:
            self._buffer.append_batch(observations)
        if self.check_period() and self.iter_num >= self.config.start:
            last_layer_name, _ = list(model.named_modules())[-1]
            observations_batch = self._buffer.sample(batch_size=self.config.batch_size)
            self.check_mont_carlo_dropout_uncertainty(
                model, observations_batch, last_layer_name
            )

    def check_mont_carlo_dropout_uncertainty(
        self, model, observations, last_layer_name
    ):
        """
        Performs Monte Carlo dropout to measure the uncertainty. A normal learning process consists of starting from
        a high uncertainty value to a low value.

        Args:
            model (nn.Module): the main model
            observations (Tensor): the observation collected after doing one-step
            last_layer_name (String): The last layer name

        Returns:

        """
        predictions = []
        mcd_model = DropoutWrapper(model, last_layer_name)
        # perform Monte Carlo dropout on the wrapped model
        for i in range(self.config.num_repetitions):
            output = mcd_model(observations)
            predictions.append(output)

        # Average the predictions to get the final prediction and uncertainty estimate
        uncertainty = torch.mean(torch.std(torch.stack(predictions), dim=0), dim=0)
        if torch.any(uncertainty > self.config.std_threshold):
            self.error_msg.append(
                self.main_msgs["mcd_uncertainty"].format(
                    torch.mean(uncertainty),
                    self.config.std_threshold,
                    self.config.num_repetitions,
                )
            )
        return None
