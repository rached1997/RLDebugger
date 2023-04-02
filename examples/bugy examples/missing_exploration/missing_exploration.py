# https://stackoverflow.com/questions/47750291/deep-q-score-stuck-at-9-for-cartpole

from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gym
import matplotlib.pyplot as plt
from time import time

from debugger import rl_debugger

t = int(time())


class DQNAgent(nn.Module):

    def __init__(self, state_size, action_size):
        super(DQNAgent, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.learning_rate = 0.001
        self.model, self.optimizer = self._build_model()

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.Tanh(),
            nn.Linear(24, 24),
            nn.Tanh(),
            nn.Linear(24, 24),
            nn.Tanh(),
            nn.Linear(24, self.action_size)
        )
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        return model, optimizer

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_values = self.model(state)
            rl_debugger.debug(actions_probs=action_values)
        return np.argmax(action_values.numpy())

    def replay(self, batch_size):
        try:
            minibatch = random.sample(self.memory, batch_size)
        except ValueError:
            minibatch = self.memory
        for state, action, reward, next_state, done in minibatch:
            state = torch.from_numpy(state).float().unsqueeze(0)
            next_state = torch.from_numpy(next_state).float().unsqueeze(0)
            action = torch.tensor(action).unsqueeze(0)
            reward = torch.tensor(reward).unsqueeze(0)
            done = torch.tensor(done).unsqueeze(0)
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model(next_state))
            q_values = self.model(state)
            q_values[0][action] = target
            rl_debugger.debug(model=self.model,
                              actions=action,
                              training_observations=state,
                              predicted_next_vals=self.model(state),
                              steps_rewards=reward,
                              steps_done=done,
                              loss_fn=nn.MSELoss(),
                              opt=self.optimizer)
            loss = nn.MSELoss()(q_values, self.model(state))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


if __name__ == "__main__":
    environment = 'CartPole-v0'
    env = gym.make(environment)
    avgs = deque(maxlen=50)
    rewardLA = []
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
    # episodes = 10000
    episodes = 2000
    rl_debugger.set_config(config_path="debugger.yml")
    rl_debugger.debug(environment=env, max_steps_per_episode=500, max_total_steps=episodes * 500,
                      max_reward=env.spec.reward_threshold)

    rewardL = []
    for e in range(episodes):
        state = env.reset()
        for time_t in range(500):
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        avgs.append(time_t)
        rewardLA.append(sum(avgs) / len(avgs))
        print("episode: ", e, "score: ", time_t)
        rewardL.append(time_t)
        agent.replay(32)
    # pickle.dump(rewardL, open(environment + "_" + str(t) + "_rewardL.pickle", "wb"))
    plt.plot(rewardLA)
