import copy
import statistics

from debugger.debugger_interface import DebuggerInterface
import torch
import numpy as np
from debugger.utils.utils import estimate_fluctuation_rmse, get_data_slope

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
        "period": 50,
        "start": 10,
        "skip_run_threshold": 10,
        "exploration_perc": 0.2,
        "exploitation_perc": 0.8,
        "low_start": {"disabled": False, "start": 3, "entropy_min_thresh": 0.3},
        "monotonicity": {"disabled": False, "increase_thresh": 0.1, "stagnation_thresh": 1e-3},
        "strong_decrease": {"disabled": False, "strong_decrease_thresh": 5, "acceleration_points_ratio": 0.2},
        "fluctuation": {"disabled": False, "fluctuation_thresh": 0.5},
        "action_stag": {"disabled": False, "start": 100, "similarity_pct_thresh": 0.8},
        "action_stag_per_ep": {"disabled": False, "nb_ep_to_check": 2, "last_step_num": 50, "reward_tolerance": 0.5}
    }
    return config


class ActionCheck(DebuggerInterface):
    """
    # Performs mulitple checks on the actions taken by the agent during the learning process
    """

    def __init__(self):
        super().__init__(check_type="Action", config=get_config())
        self._action_prob_buffer = torch.tensor([], device='cuda')
        self._entropies = torch.tensor([], device='cuda')
        self._action_buffer = []
        self._end_episode_indices = []
        self.episodes_rewards = []

    def run(self, actions_probs, max_total_steps, reward, max_reward):
        """
        This function performs the following checks on the actions:
         - During the exploration phase (in the first "exploration_perc" percentage of the total steps):
            (1) Checks whether the entropy of the actions starts with a low value.
            (2) Checks whether the entropy has decreased very quickly.
            (3) Checks whether the entropy is increasing.
            (4) Checks whether the entropy is stagnating.
        - During the exploitation (after "exploitation_perc" percentage of the total steps)):
            (1) Checks whether the sequence of actions taken in multiple episodes is stagnating (i.e. the sequence of
            actions of multiple episodes are similar).
            (2) Checks whether the actions taken within a single episode are stagnating.
        - Checks whether the entropy of the actions is fluctuating

        Args:
            actions_probs: The predictions of the model on a batch of observations.
            max_total_steps: The maximum total number of steps to finish the training.
            reward: The cumulative reward collected during one episode.
            max_reward: The reward threshold before the task is considered solved.
        """
        if self.is_final_step():
            self.episodes_rewards += [reward]
        if self.skip_run(self.config['skip_run_threshold']):
            return
        actions_probs = copy.copy(actions_probs)
        if actions_probs.dim() < 2:
            actions_probs = actions_probs.reshape((1, -1))
        if not torch.allclose(torch.sum(actions_probs, dim=1), torch.ones(actions_probs.shape[0], device='cuda')):
            actions_probs = torch.softmax(actions_probs, dim=1)
        self._action_prob_buffer = torch.cat((self._action_prob_buffer, actions_probs), dim=0)
        if self.check_period():
            entropy = self.compute_entropy()
            self._entropies = torch.cat((self._entropies, entropy.view(1)))
            self.wandb_metrics = {'entropy': entropy}
            self._action_prob_buffer = torch.tensor([], device='cuda')
            # start checking entropy of action probs
            self.check_entropy_start_very_low()
            if len(self._entropies) >= self.config['start']:
                entropy_slope = get_data_slope(self._entropies)
                if self.step_num <= max_total_steps * self.config['exploration_perc']:
                    self.check_entropy_monotonicity(entropy_slope=entropy_slope)
                    self.check_entropy_decrease_very_fast()
                self.check_entropy_fluctuation(entropy_slope=entropy_slope)
        # start checking action stagnation
        if self.step_num > max_total_steps * self.config['exploitation_perc']:
            self._action_buffer.append(torch.argmax(actions_probs).item())
            if self.check_period():
                self.check_action_stagnation_overall()
            if self.is_final_step():
                self._end_episode_indices.append(len(self._action_buffer))
                self.check_action_stagnation_per_episode(max_reward)

    def compute_entropy(self):
        """
        Computes the entropy of the action probabilities. self._action_prob_buffer is  tensor containing the action
        probabilities of shape (batch_size, num_actions)

        Returns: entropy (torch.Tensor): A scalar tensor containing the average entropy of the action probabilities.
        """
        log_probs = torch.log(self._action_prob_buffer)
        entropy = -torch.mean(torch.sum(self._action_prob_buffer * log_probs, dim=1))
        return entropy

    def check_entropy_start_very_low(self):
        """
        Checks whether the entropy of the current action's probability distribution starts with a value that is
        considered too low (less than self.config["low_start"]["entropy_min_thresh"]). A low entropy value at the start
        of the exploration process may limit the randomness of the chosen actions, leading to less efficient
        exploration.
        """
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
        Check if the entropy is increasing with time, or is stagnated during the exploration.

        Args:
            entropy_slope (float): The slope of the linear regression fit to the entropy values.
        """
        entropy_slope_cof = entropy_slope[0, 0]
        if self.config["monotonicity"]["disabled"]:
            return
        if entropy_slope_cof > self.config["monotonicity"]["increase_thresh"]:
            self.error_msg.append(self.main_msgs['entropy_incr'].format(entropy_slope_cof))
        elif torch.abs(entropy_slope_cof) < self.config["monotonicity"]["stagnation_thresh"]:
            self.error_msg.append(self.main_msgs['entropy_stag'].format(entropy_slope_cof))
        return None

    def check_entropy_decrease_very_fast(self):
        """
        Checks if the second derivative of entropy with respect to time is negative,
        which would indicate that entropy is decreasing very fast. The second derivative can tell us if the rate of
        change of the function is increasing or decreasing. In business, for example, the first derivative might tell
        us that our profits are increasing, but the second derivative will tell us if the pace of the increase is
        increasing or decreasing.
        """
        if self.config["strong_decrease"]["disabled"]:
            return
        entropy_values = self._entropies.detach().cpu().numpy()
        second_derivative = np.gradient(np.gradient(entropy_values))
        acceleration_ratio = np.mean(second_derivative < self.config["strong_decrease"]["strong_decrease_thresh"])
        if acceleration_ratio >= self.config["strong_decrease"]["acceleration_points_ratio"]:
            self.error_msg.append(self.main_msgs['entropy_strong_dec'].format(
                self.config["strong_decrease"]["strong_decrease_thresh"]))
        return None

    def check_entropy_fluctuation(self, entropy_slope):
        """
        Checks whether the entropy values of the actions are fluctuating. An unstable learning process can be indicated
        by fluctuating entropy values, as the normal behavior of entropy is to start from a high value and gradually
        decrease to zero as learning progresses.

        Args:
            entropy_slope (float): The slope of the linear regression fit to the entropy values.
        """
        if self.config["fluctuation"]["disabled"]:
            return
        residuals = estimate_fluctuation_rmse(entropy_slope, self._entropies)
        if residuals > self.config['fluctuation']["fluctuation_thresh"]:
            self.error_msg.append(self.main_msgs['entropy_fluctuation'].format(residuals, self.config["fluctuation"][
                "fluctuation_thresh"]))
        return None

    def check_action_stagnation_overall(self):
        """
        Checks whether the agent's chosen actions are stagnating, meaning that the agent consistently takes the same
        action throughout an episode.
        """
        if self.config["action_stag"]["disabled"]:
            return

        actions_tensor = torch.tensor(self._action_buffer, device="cuda")
        if len(self._action_buffer) >= self.config['action_stag']['start']:
            mode_tensor = torch.mode(actions_tensor).values

            num_matching = sum(actions_tensor == mode_tensor)
            similarity_pct = num_matching / len(self._action_buffer)

            if similarity_pct > self.config['action_stag']["similarity_pct_thresh"]:
                self.error_msg.append(self.main_msgs['action_stagnation'].format(similarity_pct))
        return None

    def check_action_stagnation_per_episode(self, max_reward):
        """
        Compares the actions taken in multiple episodes during the exploitation phase of the learning process and
        checks whether the agent is repeating the same sequence of actions. This check can help detect when the agent
        is stuck in a local optima or exhibiting erroneous behavior, such as the "noisy TV problem".
        Note that this check is only performed when the average reward is far from the maximum reward threshold.

        Args:
            max_reward:  The reward threshold before the task is considered solved
        """
        if self.config["action_stag_per_ep"]["disabled"]:
            return
        if (len(self._end_episode_indices) >= self.config['action_stag_per_ep']['nb_ep_to_check']) and \
                ((len(self.episodes_rewards) == 0) or (statistics.mean(self.episodes_rewards) < max_reward *
                                                       self.config["action_stag_per_ep"][
                                                           "reward_tolerance"])):
            final_actions = []
            for i in self._end_episode_indices:
                start_index = i - self.config["action_stag_per_ep"]["last_step_num"]
                final_actions.append(self._action_buffer[start_index:i])
            if all((final_actions[i] == final_actions[i + 1]) for i in range(len(final_actions) - 1)):
                self.error_msg.append(self.main_msgs['actions_are_similar'])
            self._end_episode_indices = []
        return None
