import copy
import statistics

from debugger.config_data_classes.rl_checkers.action_config import ActionConfig
from debugger.debugger_interface import DebuggerInterface
import torch
import numpy as np
from debugger.utils.utils import estimate_fluctuation_rmse, get_data_slope


class ActionCheck(DebuggerInterface):
    """
    This class performs checks on the actions taken by the agent during the learning process to detect abnormal
    behaviors that may hinder the training process.
    For more details on the specific checks performed, refer to the `run()` function.
    """

    def __init__(self):
        """
        Initializes the following parameters:
           * _action_buffer (list): The buffer collecting the list of actions taken in each step.
           * _action_prob_buffer (list): The buffer collecting the actions probabilities collected during training.
           * _entropies (list): The list collecting the action entropies measured each period.
           * _episodes_rewards (list): The list collecting the rewards of each episode. This list is useful to avoid
                checking action stagnation per episode when the agent reaches or is close to reaching its goal.
           * _end_episode_indices (list): This parameter marks the index of the actions in the _action_buffer that
                represent the final actions in an episode. This helps to delimit the actions taken during one episode.
        """
        super().__init__(check_type="Action", config=ActionConfig)
        self._action_buffer = []
        self._action_prob_buffer = torch.tensor([], device=self.device)
        self._entropies = torch.tensor([], device=self.device)
        self._episodes_rewards = []
        self._end_episode_indices = []

    def run(self, actions_probs, max_total_steps, reward, max_reward):
        """
        -----------------------------------   I. Introduction of the Actions Check  -----------------------------------

        I. The goal of this class is to perform multiple checks on the actions taken to detect abnormal behaviors that
        can occur during the training process.

            - Checks on the actions are primarily based on analyzing the behavior of the entropy, which measures the
            randomness of the actions taken by the agent. The normal behavior of the actions' entropy is as follows:

                * During exploration, the entropy of the actions should start from a high value as the agent should
                initially take random actions. The randomness should gradually decrease during the exploration until
                the agent stops exploring and starts the exploitation phase. Abnormal behaviors in the entropy during
                exploration can include low initial values, sharp drops, stagnation, increase instead of a decrease
                in entropy, or a fluctuating entropy during the training process which can be a sign of an unstable
                learning process.

            - In addition to the entropy, another factor that can indicate a problem in the learning process or the
            interactions between the DRL components is stagnation in the sequence of actions taken. Stagnation can take
            two main forms:

                * The stagnation can be within one episode, meaning that the agent consistently takes the same action
                throughout an episode.
                * The stagnation can occur in the sequence of actions taken in multiple successive episodes. For
                example, if an agent's possible actions are up/down, and we find that in each episode,
                the agent takes the sequence (up, up, down, up), this can be a behavioral error.

        II. The action check function performs the following checks:
                 * During the exploration phase (in the first "exploration_perc" percentage of the total steps):
                    (1) Checks whether the entropy of the actions starts with a low value.
                    (2) Checks whether the entropy has decreased very quickly.
                    (3) Checks whether the entropy is increasing.
                    (4) Checks whether the entropy is stagnating.

                 * During the exploitation phase (after "exploitation_perc" percentage of the total steps):

                    (5) Checks whether the sequence of actions taken in multiple episodes is stagnating (i.e.,
                    the sequence of actions in multiple episodes is similar).
                    (6) Checks whether the actions taken within a single episode are stagnating.

                *(7) Checks whether the entropy of the actions is fluctuating.


        III. The potential root causes behind the warnings that can be detected are:
            - Missing Exploration (checks triggered: 1..7)
            - Suboptimal exploration rate (checks triggered: 1..7)
            - The agent is stuck in a local optimum (checks triggered: 5,6)
            - Noisy tv problem (https://arxiv.org/abs/1810.12894#) (checks triggered: 5,6)
            - Bad conception of the environment (checks triggered: 1..7)
                . For example, the environment is not returning the right rewards or states


        IV. The recommended fixes for the detected issues:
            - Check whether you are doing the exploration (checks that can be fixed: 1..7)
            - Increase the exploration (1..7)
            - Check if the environment is doing the stepping correctly (checks that can be fixed: 1..7)
            - Change the architecture of the agent (checks that can be fixed: 1..7)
            - Adjust the agent's parameters:
                * Increase the batch size (checks that can be fixed: 5,6)
                * Decrease the learning rate (checks that can be fixed: 5,6)
                * Add more regularization (checks that can be fixed: 5,6)
                * Use a different optimizer (checks that can be fixed: 5,6)
                * Increase the network size (checks that can be fixed: 1..7)
                * Use a target network if possible (checks that can be fixed: 5,6)

        Examples
        --------
        To perform action checks, the debugger needs to be called after the RL agent has predicted the action.

        >>> from debugger import rl_debugger
        >>> ...
        >>> action, action_logprob, state_val, action_probs = policy_old.act(state)
        >>> rl_debugger.debug(actions_probs=action_probs, max_total_steps=max_total_steps, max_reward=max_reward)

        Note that you don't need to pass the reward to the 'debug()' function as it's automatically observed by the
        debugger.
        In the context of DQN, the act() method is the ideal location to invoke the debugger to perform action checks.

        >>> from debugger import rl_debugger
        >>> ...
        >>> state, reward, done, _ = env.step(action)
        >>> qvals = qnet(state)
        >>> rl_debugger.debug(actions_probs=qvals.detach(), max_total_steps=max_total_steps, max_reward=max_reward)

        If you feel that this check is slowing your code, you can increase the value of "skip_run_threshold" in
        ActionConfig.

        Args:
            actions_probs: The predictions of the model on a batch of observations.
            max_total_steps: The maximum total number of steps to finish the training.
            reward: The cumulative reward collected during one episode.
            max_reward: The reward threshold before the task is considered solved.
        """
        # start checking action stagnation
        if self.step_num > max_total_steps * self.config.exploitation_perc:
            self._action_buffer.append(torch.argmax(actions_probs).item())
            if self.is_final_step():
                self._episodes_rewards += [reward]
                self._end_episode_indices.append(len(self._action_buffer))
                self.check_action_stagnation_per_episode(max_reward)
            if self.check_period():
                self.check_action_stagnation_overall()

        if self.skip_run(self.config.skip_run_threshold):
            return
        actions_probs = copy.deepcopy(actions_probs)
        if actions_probs.dim() < 2:
            actions_probs = actions_probs.reshape((1, -1))
        if not torch.allclose(
            torch.sum(actions_probs, dim=1),
            torch.ones(actions_probs.shape[0], device=self.device),
        ):
            actions_probs = torch.softmax(actions_probs, dim=1)
        self._action_prob_buffer = self._action_prob_buffer.to(self.device)
        self._action_prob_buffer = torch.cat(
            (self._action_prob_buffer, actions_probs), dim=0
        )
        if self.check_period():
            entropy = self.compute_entropy()
            self._entropies = self._entropies.to(self.device)
            self._entropies = torch.cat((self._entropies, entropy.view(1)))
            self.wandb_metrics = {"entropy": entropy}
            self._action_prob_buffer = torch.tensor([], device=self.device)
            # start checking entropy of action probs
            self.check_entropy_start_very_low()
            if len(self._entropies) >= self.config.start:
                entropy_slope = get_data_slope(self._entropies)
                if self.step_num <= max_total_steps * self.config.exploration_perc:
                    self.check_entropy_monotonicity(entropy_slope=entropy_slope)
                    self.check_entropy_decrease_very_fast()
                self.check_entropy_fluctuation(entropy_slope=entropy_slope)

    def compute_entropy(self):
        """
        Computes the entropy of the action probabilities. The entropy formula is defined as:
            . H(p) = - sum(p_i * log2(p_i)) for i in 1 to n,
        where p_i is the probability of the ith action and n is the total number of actions.


        Returns: entropy (torch.Tensor): A scalar tensor containing the average entropy of the action probabilities.
        """
        log_probs = torch.log(self._action_prob_buffer)
        entropy = -torch.mean(torch.sum(self._action_prob_buffer * log_probs, dim=1))
        return entropy

    def check_entropy_start_very_low(self):
        """
        Checks whether the entropy of the current action's probability distribution starts with a value that is
        considered too low (less than self.config.low_start.entropy_min_thresh). A low entropy value at the start
        of the exploration process may limit the randomness of the chosen actions, leading to less efficient
        exploration.
        """
        if self.config.low_start.disabled:
            return
        # Check for very low mean and standard deviation of entropy
        if len(self._entropies) == self.config.low_start.start:
            mean = torch.mean(self._entropies)
            if mean < self.config.low_start.entropy_min_thresh:
                self.error_msg.append(self.main_msgs["entropy_start"].format(mean))
            self.config.low_start.disabled = True
        return None

    def check_entropy_monotonicity(self, entropy_slope):
        """
        Check if the entropy is increasing with time, or is stagnated during the exploration.

        Args:
            entropy_slope : The slope of the linear regression fit to the entropy values.
        """
        entropy_slope_cof = entropy_slope[0, 0]
        if self.config.monotonicity.disabled:
            return
        if entropy_slope_cof > self.config.monotonicity.increase_thresh:
            self.error_msg.append(
                self.main_msgs["entropy_incr"].format(entropy_slope_cof)
            )
        elif torch.abs(entropy_slope_cof) < self.config.monotonicity.stagnation_thresh:
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
        if self.config.strong_decrease.disabled:
            return
        entropy_values = self._entropies.detach().cpu().numpy()
        second_derivative = np.gradient(
            np.gradient(entropy_values[-self.config.strong_decrease.region_length :])
        )
        acceleration_ratio = np.mean(
            second_derivative < self.config.strong_decrease.strong_decrease_thresh
        )
        if acceleration_ratio >= self.config.strong_decrease.acceleration_points_ratio:
            self.error_msg.append(
                self.main_msgs["entropy_strong_dec"].format(
                    self.config.strong_decrease.strong_decrease_thresh
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
        if self.config.fluctuation.disabled:
            return
        residuals = estimate_fluctuation_rmse(
            entropy_slope, self._entropies[-self.config.fluctuation.region_length :]
        )
        if residuals > self.config.fluctuation.fluctuation_thresh:
            self.error_msg.append(
                self.main_msgs["entropy_fluctuation"].format(
                    residuals, self.config.fluctuation.fluctuation_thresh
                )
            )
        return None

    def check_action_stagnation_overall(self):
        """
        Checks whether the agent's chosen actions are stagnating, meaning that the agent consistently takes the same
        action throughout an episode.
        """
        if self.config.action_stag.disabled:
            return

        actions_tensor = torch.tensor(self._action_buffer, device=self.device)
        if len(self._action_buffer) >= self.config.action_stag.start:
            mode_tensor = torch.mode(actions_tensor).values

            num_matching = sum(actions_tensor == mode_tensor)
            similarity_pct = num_matching / len(self._action_buffer)

            if similarity_pct > self.config.action_stag.similarity_pct_thresh:
                self.error_msg.append(
                    self.main_msgs["action_stagnation"].format(similarity_pct * 100)
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
        if self.config.action_stag_per_ep.disabled:
            return
        if (
            len(self._end_episode_indices)
            >= self.config.action_stag_per_ep.nb_ep_to_check
        ) and (
            (len(self._episodes_rewards) == 0)
            or (
                statistics.mean(self._episodes_rewards)
                < max_reward * self.config.action_stag_per_ep.reward_tolerance
            )
        ):
            final_actions = []
            for i in self._end_episode_indices:
                start_index = i - self.config.action_stag_per_ep.last_step_num
                final_actions.append(self._action_buffer[start_index:i])
            if all(
                (final_actions[i] == final_actions[i + 1])
                for i in range(len(final_actions) - 1)
            ):
                self.error_msg.append(
                    self.main_msgs["actions_are_similar"].format(
                        len(self._end_episode_indices)
                    )
                )
            self._end_episode_indices = []
        return None
