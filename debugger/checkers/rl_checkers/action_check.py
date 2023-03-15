import copy
import statistics

from debugger.config_data_classes.rl_checkers.action_config import ActionConfig
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


class ActionCheck(DebuggerInterface):
    """
    This function performs multiple checks on the actions taken by the agent during the learning process. You can check
    the functions run for more details on the checks that are done by this function.

    """

    def __init__(self):
        """
        Initilizes the following parameters:
        _action_buffer : the buffer collecting the list of actions taken in each step
        _action_prob_buffer : the buffer collecting the actions probabilities collected during the training.
        _entropies : the list collecting the action entropies measured each period
        episodes_rewards : The list collecting the rewards of each episode (This list is useful to avoid doing the
        check_action_stagnation_per_episode when the agent reaches or is close to reach its goal)
        _end_episode_indices : this parameter is to mark the index of the actions in the
        _action_buffer : Represents the indexes final actions in an episode. This would help us delimite the actions taken during one episode
        """
        super().__init__(check_type="Action", config=ActionConfig)
        self._action_buffer = []
        self._action_prob_buffer = torch.tensor([], device=self.device)
        self._entropies = torch.tensor([], device=self.device)
        self.episodes_rewards = []
        self._end_episode_indices = []

    def run(self, actions_probs, max_total_steps, reward, max_reward):
        """
        The goal of this function is to do multiple checks on the actions taken to catch abnormal behaviours that can
        occure during the training process

        - The checks on the actions are mainly based on analysing the beahviour of the entropy. The entropy consists of
        analysing the randomness of the actions taken by the agent. The normal behaviour of the actions entropy should be as following:

            * During the exploration the entropy of the actions should first start from a high value as during the
            exploration the agent should start by taking random actions than the randomness should keep on decreasing
            gradually during the exploration until the agent stops exploring and starts the exploitation fase. The wrong
            behaviours that can occure in the entropy behaviour during the exploration are to find that the entropystarts
            with a low value, detect if there is a sharp drop in the entropy valu, the entropy value is stagnating and
            the entropy value is increasing instead of decreasing

            * In addition one wrong behaviour that can be detected during the whole training process of the agent is a
            fluctuating entropy. A fluctuation in the entropy would generally be a sign of an unstable learning process.

        In addtion to the entropy, another factor that can show a problem in the learning process or in the
        interactions between the DRL components is the stagnation in the sequence of actions taken. The stagnation
        can have two main forms:

            * The stagnation can be in one episode which means that the agent is consistently taking the same action
            throughout an episode

            * The stagnation can be in the sequence of actions taken in multiple successive episodes. For example if
            an agent's possible actions are (up/ down) and we find that in each episode the agent takes the sequence
            (up,up,down,up) this can be a behavioural error

        The action check performs the following :
         - During the exploration phase (in the first "exploration_perc" percentage of the total steps) it does the the wrong behaviours that the action check can detect are:
            (1) Checks whether the entropy of the actions starts with a low value.
            (2) Checks whether the entropy has decreased very quickly.
            (3) Checks whether the entropy is increasing.
            (4) Checks whether the entropy is stagnating.
        - During the exploitation (after "exploitation_perc" percentage of the total steps)):
            (5) Checks whether the sequence of actions taken in multiple episodes is stagnating (i.e. the sequence of
            actions of multiple episodes are similar).
            (6) Checks whether the actions taken within a single episode are stagnating.
        - (7) Checks whether the entropy of the actions is fluctuating


        The potential root causes behind the warnings that can be detected are
            - Missing Exploration (checks triggered : 1..7)
            - Suboptimal exploration rate (checks triggered : 1..7)
            - The agent is stuck in a local optimum (checks triggered : 5,6)
            - Noisy tv problem (checks triggered : 5,6)
            - Bad conception of the environment
                - For example the environment is not returning the right rewards or states (checks triggered : 1..7)

        The recommended fixes for the detected issues:
        - Check whether you are doing the exploration ( checks tha can be fixed: 1..7)
        - Increase the exploration (1..7)
        - Check if the environment is doing the stepping correctly ( checks tha can be fixed: 1..7)
        - Change the architecture of the agent (checks tha can be fixed: 1..7)
        - Adjust the agent's parameters
            - Increase the batch size ( checks tha can be fixed: 5,6)
            - Decrease the learning rate (checks tha can be fixed: 5,6)
            - Add more regularization (checks tha can be fixed: 5,6)
            - Use a different optimizer (checks tha can be fixed: 5,6)
            - Increase the network size (checks tha can be fixed: 1..7)
            - Use a target network if possible (checks tha can be fixed: 5..6)


        Args:
            actions_probs: The predictions of the model on a batch of observations.
            max_total_steps: The maximum total number of steps to finish the training.
            reward: The cumulative reward collected during one episode.
            max_reward: The reward threshold before the task is considered solved.
        """
        if self.is_final_step():
            self.episodes_rewards += [reward]
        if self.skip_run(self.config["skip_run_threshold"]):
            return
        actions_probs = copy.copy(actions_probs)
        if actions_probs.dim() < 2:
            actions_probs = actions_probs.reshape((1, -1))
        if not torch.allclose(
            torch.sum(actions_probs, dim=1),
            torch.ones(actions_probs.shape[0], device=self.device),
        ):
            actions_probs = torch.softmax(actions_probs, dim=1)
        self._action_prob_buffer = torch.cat(
            (self._action_prob_buffer, actions_probs), dim=0
        )
        if self.check_period():
            entropy = self.compute_entropy()
            self._entropies = torch.cat((self._entropies, entropy.view(1)))
            self.wandb_metrics = {"entropy": entropy}
            self._action_prob_buffer = torch.tensor([], device=self.device)
            # start checking entropy of action probs
            self.check_entropy_start_very_low()
            if len(self._entropies) >= self.config["start"]:
                entropy_slope = get_data_slope(self._entropies)
                if self.step_num <= max_total_steps * self.config["exploration_perc"]:
                    self.check_entropy_monotonicity(entropy_slope=entropy_slope)
                    self.check_entropy_decrease_very_fast()
                self.check_entropy_fluctuation(entropy_slope=entropy_slope)
        # start checking action stagnation
        if self.step_num > max_total_steps * self.config["exploitation_perc"]:
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
                self.error_msg.append(self.main_msgs["entropy_start"].format(mean))
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
            self.error_msg.append(
                self.main_msgs["entropy_incr"].format(entropy_slope_cof)
            )
        elif (
            torch.abs(entropy_slope_cof)
            < self.config["monotonicity"]["stagnation_thresh"]
        ):
            self.error_msg.append(
                self.main_msgs["entropy_stag"].format(entropy_slope_cof)
            )
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
        acceleration_ratio = np.mean(
            second_derivative < self.config["strong_decrease"]["strong_decrease_thresh"]
        )
        if (
            acceleration_ratio
            >= self.config["strong_decrease"]["acceleration_points_ratio"]
        ):
            self.error_msg.append(
                self.main_msgs["entropy_strong_dec"].format(
                    self.config["strong_decrease"]["strong_decrease_thresh"]
                )
            )
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
        if residuals > self.config["fluctuation"]["fluctuation_thresh"]:
            self.error_msg.append(
                self.main_msgs["entropy_fluctuation"].format(
                    residuals, self.config["fluctuation"]["fluctuation_thresh"]
                )
            )
        return None

    def check_action_stagnation_overall(self):
        """
        Checks whether the agent's chosen actions are stagnating, meaning that the agent consistently takes the same
        action throughout an episode.
        """
        if self.config["action_stag"]["disabled"]:
            return

        actions_tensor = torch.tensor(self._action_buffer, device=self.device)
        if len(self._action_buffer) >= self.config["action_stag"]["start"]:
            mode_tensor = torch.mode(actions_tensor).values

            num_matching = sum(actions_tensor == mode_tensor)
            similarity_pct = num_matching / len(self._action_buffer)

            if similarity_pct > self.config["action_stag"]["similarity_pct_thresh"]:
                self.error_msg.append(
                    self.main_msgs["action_stagnation"].format(similarity_pct)
                )
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
        if (
            len(self._end_episode_indices)
            >= self.config["action_stag_per_ep"]["nb_ep_to_check"]
        ) and (
            (len(self.episodes_rewards) == 0)
            or (
                statistics.mean(self.episodes_rewards)
                < max_reward * self.config["action_stag_per_ep"]["reward_tolerance"]
            )
        ):
            final_actions = []
            for i in self._end_episode_indices:
                start_index = i - self.config["action_stag_per_ep"]["last_step_num"]
                final_actions.append(self._action_buffer[start_index:i])
            if all(
                (final_actions[i] == final_actions[i + 1])
                for i in range(len(final_actions) - 1)
            ):
                self.error_msg.append(self.main_msgs["actions_are_similar"])
            self._end_episode_indices = []
        return None
