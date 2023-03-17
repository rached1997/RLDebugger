import statistics
from debugger.config_data_classes.rl_checkers.steps_config import StepsConfig
from debugger.debugger_interface import DebuggerInterface


class StepCheck(DebuggerInterface):
    """
    This class performs checks on the maximum number of steps that can be reached .
    For more details on the specific checks performed, refer to the `run()` function.
    """
    def __init__(self):
        """
        Initializes the following parameters:
            * _final_step_number_buffer :  list storing the number of steps taken in each episode
            * _episode_reward_buffer : list storing the rewards accumulated in each episode
            * _last_step_num : an integetr storing the total number of steps the agent did in the previous episode
        """
        super().__init__(check_type="Step", config=StepsConfig)
        self._final_step_number_buffer = []
        self._episode_reward_buffer = []
        self._last_step_num = 0

    def run(self, reward, max_reward, max_total_steps, max_steps_per_episode) -> None:
        """
        -----------------------------------   I. Introduction of the Step Check   -----------------------------------
        The number of steps represents the number of times that the agent interacts with its environment. In DRL,
        every episode is defined by a maximum number of steps, and once that limit is reached, the experiment starts
        over. It's essential to set an appropriate value for the maximum number of steps per episode to avoid having
        episodes that end prematurely, which can reduce the learning efficiency of the agent.

        The primary goal of this class is to ensure that episodes are not being ended prematurely due to the maximum
        step limit being reached, especially during the exploitation phase when the agent is not learning,
        and the reward is far from the maximum reward.

        ------------------------------------------   II. The performed checks  -----------------------------------------

        The steps check performs the following check:
            (1) Checks whether the max steps per episode has a low value.

        ------------------------------------   III. The potential Root Causes  -----------------------------------------

        The potential root causes behind the warnings that can be detected are:
            - The max steps per episode has a low value (checks triggered : 1)

        --------------------------------------   IV. The Recommended Fixes  --------------------------------------------

        The recommended fixes for the detected issues :
            - Increase the max step number per episode value (checks that can be fixed: 1)

        Examples
        --------
        To perform step checks, the debugger needs "max_reward", "max_total_steps" and "max_steps_per_episode"
        parameters only (constant parameters). The 'reward' parameter is automatically observed by debugger, and you
        don't need to pass it to the 'debug()' function.
        The best way to run these checks is to provide the "max_reward", "max_total_steps" and "max_steps_per_episode"
        at the begging of your code.

        >>> from debugger import rl_debugger
        >>> ...
        >>> env = gym.make("CartPole-v1")
        >>> rl_debugger.debug(max_reward=max_reward, max_total_steps=max_total_steps,
        >>>                   max_steps_per_episode=max_steps_per_episode)

        If you feel that these checks are slowing your code, you can increase the value of "skip_run_threshold" in
        RewardConfig.

        Args:
            reward (float): the cumulative reward collected in one episode
            max_reward (int):  The reward threshold before the task is considered solved
            max_total_steps (int): The maximum total number of steps to finish the training.
            max_steps_per_episode (int): the max steps for an episode
        """
        if self.is_final_step():
            self._final_step_number_buffer += [self.step_num - self._last_step_num]
            self._episode_reward_buffer += [reward]
            self._last_step_num = self.step_num
        # TODO: check if this change is working
        if self.skip_run(self.config.skip_run_threshold):
            return

        self.check_step_is_not_changing(
            max_reward, max_total_steps, max_steps_per_episode
        )

    def check_step_is_not_changing(
        self, max_reward, max_total_steps, max_steps_per_episode
    ):
        """
        Checks if episodes are being ended prematurely due to the max step limit being reached during the
        exploitation phase when the agent is not learning (i.e. the reward is far from the max reward).

        Args:
            max_reward (int):  The reward threshold before the task is considered solved
            max_total_steps (int): The maximum total number of steps to finish the training.
            max_steps_per_episode (int): the max steps for an episode
        """
        if self.config.check_stagnation.disabled:
            return

        if self.check_period() and (
            self.step_num >= (max_total_steps * self.config.exploitation_perc)
        ):
            if (
                    statistics.mean(self._final_step_number_buffer) >= max_steps_per_episode
            ) and (
                statistics.mean(self._episode_reward_buffer)
                < (max_reward * self.config.poor_max_step_per_ep.max_reward_tol)
            ):
                self.error_msg.append(self.main_msgs["poor_max_step_per_ep"])
