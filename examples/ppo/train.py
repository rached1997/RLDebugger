# Based on implementation from Barhate, Nikhil repository (https://github.com/nikhilbarhate99/PPO-PyTorch)
# Accessed on 3/11/2023
import cProfile as profile
import pstats
import time
import torch
import gym
from examples.ppo.ppo import PPO
from debugger import rl_debugger
import numpy as np

# Set seeds for Torch
torch.manual_seed(42)

# Set seeds for Torch's backend (CPU or GPU)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
else:
    np.random.seed(42)


def main():
    incr = 0
    env_name = "CartPole-v1"
    max_ep_len = 400
    max_training_timesteps = int(1e5)
    print_freq = max_ep_len * 4
    update_timestep = max_ep_len * 4
    k_epochs = 40
    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor
    lr_actor = 0.0003  # learning rate for actor network
    lr_critic = 0.001  # learning rate for critic network

    env = gym.make(env_name)
    env.seed(42)
    rl_debugger.debug(
        environment=env,
        max_reward=max_ep_len,
        max_total_steps=max_training_timesteps,
        max_steps_per_episode=max_ep_len,
    )
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    ppo_agent = PPO(
        state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs, eps_clip
    )

    start_time = time.time()
    print_running_reward = 0
    print_running_episodes = 0
    time_step = 0
    i_episode = 0

    while time_step <= max_training_timesteps:
        state = env.reset()
        current_ep_reward = 0

        for t in range(1, max_ep_len + 1):
            # select action with policy
            action = ppo_agent.select_action(state, incr)
            incr += 1
            state, reward, done, _ = env.step(action)

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # printing average reward
            # if time_step % print_freq == 0:
            #     print_avg_reward = print_running_reward / print_running_episodes
            #     print_avg_reward = round(print_avg_reward, 2)
            #     print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step,
            #                                                                             print_avg_reward))
            #     print_running_reward = 0
            #     print_running_episodes = 0

            # break; if the episode is over
            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        print_avg_reward = print_running_reward / print_running_episodes
        print_avg_reward = round(print_avg_reward, 2)
        print(
            "Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(
                i_episode, time_step, print_avg_reward
            )
        )
        print_running_reward = 0
        print_running_episodes = 0
        i_episode += 1

    env.close()

    # print total training time
    end_time = time.time()
    print("Total training time  : ", end_time - start_time)


if __name__ == "__main__":
    # perform profiling
    # prof = profile.Profile()
    # prof.enable()
    rl_debugger.set_config(config_path="debugger.yml")
    main()
    # prof.disable()
    # # print profiling output
    # stats = pstats.Stats(prof).strip_dirs().sort_stats("ncalls")
    # stats.print_stats('.*')  # top 10 rows
    # print(rl_debugger.time)
