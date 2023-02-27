from debugger.debugger_interface import DebuggerInterface
import torch
import numpy as np

"""
the entropy regularization or exploration bonus.  In the early stages of learning, the entropy of the policy 
distribution should be high to encourage exploration and to prevent the agent from becoming too deterministic. 
As the agent gains more experience and becomes more confident in its policy, the entropy can be gradually reduced 
to encourage exploitation of the learned policy.

"""


def get_config():
    """
        Return the configuration dictionary needed to run the checkers.

        Returns:
            config (dict): The configuration dictionary containing the necessary parameters for running the checkers.
    """
    config = {
        "Period": 100,
        "window_size": 20,
        "exploration_perc": 0.2,
        "low_start": {"disabled": False, "start": 5, "entropy_min_thresh": 0.3},
        "monotonicity": {"disabled": False, "increase_thresh": 0.1, "stagnation_thresh": 1e-3},
        "strong_decrease": {"disabled": False, "strong_decrease_thresh": -0.05, "acceleration_points_ratio": 0.5},
        "fluctuation": {"disabled": False, "fluctuation_thresh": 0.5}

    }
    return config


class OnTrainActionCheck(DebuggerInterface):
    """
    #
    """

    def __init__(self):
        super().__init__(check_type="OnTrainAction", config=get_config())
        self._action_prob_buffer = torch.tensor([], device='cuda')
        self._entropies = torch.tensor([], device='cuda')

    def run(self, actions_probs, max_total_steps):
        """
        #
        """
        if not torch.allclose(torch.sum(actions_probs, dim=1), torch.ones(actions_probs.shape[0], device='cuda')):
            actions_probs = torch.softmax(actions_probs, dim=1)

        self._action_prob_buffer = torch.cat((self._action_prob_buffer, actions_probs), dim=0)
        if self.check_period():
            self._entropies = torch.cat((self._entropies, self.compute_entropy().view(1)))
            self._action_prob_buffer = torch.tensor([], device='cuda')
            # start checking entropy of action probs
            self.check_entropy_start_very_low()
            if len(self._entropies) >= self.config['window_size']:
                if self.iter_num < max_total_steps * self.config['exploration_perc']:
                    self.check_entropy_monotonicity(entropy_slope=self.get_entropy_slope())
                    self.check_entropy_decrease_very_fast()
                self.check_entropy_fluctuation()

    def compute_entropy(self):
        """
            Computes the entropy of the action probabilities. self._action_prob_buffer is  tensor containing the action
            probabilities of shape (batch_size, num_actions)

            Returns: entropy (torch.Tensor): A scalar tensor containing the average entropy of the action probabilities.
            """
        log_probs = torch.log(self._action_prob_buffer)
        entropy = -torch.mean(torch.sum(self._action_prob_buffer * log_probs, dim=1))
        return entropy

    def get_entropy_slope(self):
        """Compute the slope of entropy evolution over time.

        Returns:
        entropy_slope (float): The slope of the linear regression fit to the entropy values.
        """
        # Compute the x-values (time steps) for the linear regression
        x = torch.arange(len(self._entropies), device=self._entropies.device)
        # Fit a linear regression model to the entropy values
        ones = torch.ones_like(x)
        X = torch.stack([x, ones], dim=1).float()
        cof, _ = torch.lstsq(self._entropies.unsqueeze(1), X)

        return cof[0, 0]

    def check_entropy_start_very_low(self):
        if self.config["low_start"]["disabled"]:
            return
        # Check for very low mean and standard deviation of entropy
        if len(self._entropies) == self.config["low_start"]["start"]:
            mean = torch.mean(self._entropies)
            if mean < self.config["low_start"]["entropy_min_thresh"]:
                self.error_msg.append(self.main_msgs['entropy_start'].format(mean))
        return None

    def check_entropy_monotonicity(self, entropy_slope):
        """
        Check if the entropy is increasing with time, or is stagnated.

        entropy_slope (float): The slope of the linear regression fit to the entropy values.
        :return: A warning message if the entropy is increasing or stagnated with time.
        """
        if self.config["monotonicity"]["disabled"]:
            return
        if entropy_slope > self.config["monotonicity"]["increase_thresh"]:
            self.error_msg.append(self.main_msgs['entropy_incr'].format(entropy_slope))
        elif torch.abs(entropy_slope) < self.config["monotonicity"]["stagnation_thresh"]:
            self.error_msg.append(self.main_msgs['entropy_stag'].format(entropy_slope))
        return None

    def check_entropy_decrease_very_fast(self):
        """
        Checks if the second derivative of entropy with respect to time is negative,
        which would indicate that entropy is decreasing very fast. The second derivative can tell us if the rate of
        change of the function is increasing or decreasing. In business, for example, the first derivative might tell
        us that our profits are increasing, but the second derivative will tell us if the pace of the increase is
        increasing or decreasing.

        Returns: A warning message if the entropy is decreasing very fast with time.
        """
        if self.config["strong_decrease"]["disabled"]:
            return
        # TODO: try make this torch
        entropy_values = self._entropies.detach().cpu().numpy()
        time_values = len(entropy_values)
        second_derivative = np.gradient(np.gradient(entropy_values, time_values), time_values)
        acceleration_ratio = np.mean(second_derivative < self.config["strong_decrease"]["strong_decrease_thresh"])
        if acceleration_ratio >= self.config["strong_decrease"]["acceleration_points_ratio"]:
            self.error_msg.append(self.main_msgs['entropy_strong_dec'].format(
                self.config["strong_decrease"]["strong_decrease_thresh"]))
        return None

    def check_entropy_fluctuation(self, entropy_slope):
        if self.config["fluctuation"]["disabled"]:
            return
        x = torch.arange(len(self._entropies), device=self._entropies.device)
        ones = torch.ones_like(x)
        X = torch.stack([x, ones], dim=1).float()
        predicted = X.mm(entropy_slope.solution.T)

        residuals = torch.sqrt(torch.mean((self._entropies - predicted) ** 2))
        if residuals > self.config['fluctuation']["fluctuation_thresh"]:
            # TODO: check message please
            self.error_msg.append(self.main_msgs['entropy_fluctuation'].format(residuals, self.config["fluctuation"][
                "fluctuation_thresh"]))
        return None

# TODO: check action stagnation overall
# TODO: check action stagnation per episode (periodic)
