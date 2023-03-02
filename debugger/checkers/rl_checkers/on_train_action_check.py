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
        "exploitation_perc": 0.8,
        "low_start": {"disabled": False, "start": 5, "entropy_min_thresh": 0.3},
        "monotonicity": {"disabled": False, "increase_thresh": 0.1, "stagnation_thresh": 1e-3},
        "strong_decrease": {"disabled": False, "strong_decrease_thresh": -0.05, "acceleration_points_ratio": 0.5},
        "fluctuation": {"disabled": False, "fluctuation_thresh": 0.5},
        "action_stag": {"disabled": False, "start": 100, "similarity_pct_thresh": 0.8},
        "action_stag_per_ep": {"disabled": False, "nb_ep_to_check": 2, "last_step_num": 50}

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
        self._action_buffer = torch.tensor([], device='cuda')
        self.end_episode_indices = []

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
                entropy_slope = self.get_entropy_slope()
                if self.step_num < max_total_steps * self.config['exploration_perc']:
                    self.check_entropy_monotonicity(entropy_slope=entropy_slope)
                    self.check_entropy_decrease_very_fast()
                self.check_entropy_fluctuation(entropy_slope=entropy_slope)
            # start checking action stagnation
            if self.step_num > max_total_steps * self.config['exploitation_perc']:
                self._action_buffer = torch.cat((self._action_prob_buffer, torch.argmax(actions_probs).item()), dim=0)
                self.check_action_stagnation_overall()
                if self.is_final_step_of_ep():
                    self.end_episode_indices.append(len(self._action_buffer) - 1)
                    self.check_action_stagnation_per_episode()

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
            self.error_msg.append(self.main_msgs['entropy_fluctuation'].format(residuals, self.config["fluctuation"][
                "fluctuation_thresh"]))
        return None

    def check_action_stagnation_overall(self):
        if self.config["action_stag"]["disabled"]:
            return
        if len(self._action_buffer) >= self.config['action_stagnation']['start']:
            mode_tensor = torch.mode(self._action_buffer).values

            num_matching = sum(self._action_buffer == mode_tensor)
            similarity_pct = num_matching / len(self._action_buffer)

            if similarity_pct > self.config['action_stag']["similarity_pct_thresh"]:
                self.error_msg.append(self.main_msgs['action_stagnation'].format(similarity_pct))
        return None

    def check_action_stagnation_per_episode(self):
        if self.config["action_stag_per_ep"]["disabled"]:
            return
        if len(self.end_episode_indices) >= self.config['action_stag_per_ep']['nb_ep_to_check']:
            final_obs = []
            for i in self.end_episode_indices:
                start_index = i - self.config["action_stag_per_ep"]["last_step_num"]
                final_obs += self._action_buffer[start_index:i + 1]
            if all((final_obs[i] == final_obs[i+1]) for i in range(len(final_obs)-1)):
                self.error_msg.append(self.main_msgs['observations_are_similar'])
        self.end_episode_indices = []
