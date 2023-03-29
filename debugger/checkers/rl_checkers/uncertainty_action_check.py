from debugger.config_data_classes.rl_checkers.uncertainty_action_config import (
    UncertaintyActionConfig,
)
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

    def sample(self, batch_size, device):
        indices = random.sample(range(self.size), batch_size)
        result = torch.stack([self.buffer[index] for index in indices], dim=0).squeeze()
        return torch.tensor(result, device=device)


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

    def run(self, model: torch.nn.Module, observations: torch.Tensor, environment):
        """
        Checks whether the actions predictions uncertainty over time is reducing during the learning process

        Examples
        --------
        To perform uncertainty action checks, the debugger needs to be called after the RL agent has predicted the
        action. The debugger needs "model" parameter only to perform these checks. The 'observations' parameter is
        automatically observed by debugger, and you don't need to pass it to the 'debug()' function.

        >>> from debugger import rl_debugger
        >>> ...
        >>> action, action_logprob, state_val, action_probs = policy_old.act(state)
        >>> rl_debugger.debug(model=policy_old.actor)

        In the context of DQN, the act() method is the ideal location to invoke the debugger to perform uncertainty
        action checks.

        >>> from debugger import rl_debugger
        >>> ...
        >>> state, reward, done, _ = env.step(action)
        >>> qvals = qnet(state)
        >>> rl_debugger.debug(model=qnet)

        If you feel that this check is slowing your code, you can increase the value of "skip_run_threshold" in
        UncertaintyActionConfig.

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
            observations_batch = self._buffer.sample(batch_size=self.config.batch_size, device=self.device)
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
