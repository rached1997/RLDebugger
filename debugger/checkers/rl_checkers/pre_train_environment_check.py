import hashlib
import torch
from debugger.debugger_interface import DebuggerInterface


def get_config() -> dict:
    """
    Return the configuration dictionary needed to run the checkers.

    Returns:
        config (dict): The configuration dictionary containing the necessary parameters for running the checkers.
    """
    config = {
        "Period": 0,
        "observations_std_coef_thresh": 0.001,
        "Markovianity_check": {"disabled": False, "num_trajectories": 1000}
    }
    return config


class PreTrainEnvironmentCheck(DebuggerInterface):
    def __init__(self):
        super().__init__(check_type="PreTrainEnvironment", config=get_config())
        self.obs_list = torch.tensor([], device='cuda')
        self.reward_list = torch.tensor([], device='cuda')
        self.done_list = torch.tensor([], device='cuda')
        self.info_list = torch.tensor([], device='cuda')

    def run(self, environment) -> None:

        # TODO: add Markovianity check
        if not self.config["Markovianity_check"]["disabled"]:
            pass

        # TODO: check Rest and Step function (no nan returns)
        # TODO: check observations and actions in gym box
        # TODO: check max_reward and max_step_per_episode

        # TODO: ask if rest should be random or yield the same states

        if environment.spec.max_episode_steps:
            self.generate_random_eps(environment)
            if sum(self.reward_list) > environment.spec.reward_threshold:
                self.error_msg.append(self.main_msgs['Weak_reward_threshold'])

            if torch.mean(torch.std(self.obs_list, dim=0)) <= self.config["observations_std_coef_thresh"]:
                self.error_msg.append(
                    self.main_msgs['invalid_step_func'].format(torch.mean(torch.std(self.obs_list, dim=0))))

    def generate_random_eps(self, environment):
        done = False
        initial_obs = environment.reset()
        self.obs_list = torch.cat((self.obs_list, initial_obs), dim=0)

        step = 0
        while (not done) and (step < environment.spec.max_episode_steps):
            step += 1
            obs, reward, done, info = environment.step(environment.action_space.sample())
            self.obs_list = torch.cat((self.obs_list, obs), dim=0)
            self.reward_list = torch.cat((self.reward_list, reward), dim=0)
            self.done_list = torch.cat((self.done_list, done), dim=0)
            self.info_list = torch.cat((self.info_list, info), dim=0)

    # def generate_random_trajectories(self, env):
    #     trajectories = []
    #     for i in range(self.config["Markovianity_check"]["num_trajectories"]):
    #         obs = env.reset()
    #         trajectory = []
    #         for t in range(env.spec.max_episode_steps):
    #             action = env.action_space.sample()
    #             obs_next, reward, done, info = env.step(action)
    #             hashed_obs = str(hashlib.sha256(obs.tobytes()).hexdigest())
    #             hashed_obs_next = str(hashlib.sha256(obs_next.tobytes()).hexdigest())
    #             trajectory.append((hashed_obs, action, reward, hashed_obs_next))
    #             if done:
    #                 break
    #             obs = obs_next
    #         trajectories.append(trajectory)
    #     return trajectories
    #
    # def check_markovianity(self, env):
    #     trajectories = self.generate_random_trajectories(env)
    #     is_markovian = True
    #     for trajectory in trajectories:
    #         for t in range(len(trajectory) - 1):
    #             obs_t, action_t, reward_t, obs_next_t = trajectory[t]
    #             obs_next_t_predicted = env.reset()
    #             for t_prime in range(t, len(trajectory)):
    #                 obs_t_prime, action_t_prime, reward_t_prime, obs_next_t_prime = trajectory[t_prime]
    #                 if np.array_equal(obs_t_prime, obs_t):
    #                     obs_next_t_predicted = obs_next_t_prime
    #                     break
    #             if not np.array_equal(obs_next_t, obs_next_t_predicted):
    #                 is_markovian = False
    #                 break
    #         if not is_markovian:
    #             break
