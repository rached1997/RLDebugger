# https://stackoverflow.com/questions/54385568/tensorflow-dqn-cant-solve-openai-cartpole

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from debugger import rl_debugger


class QNet(nn.Module):
    def __init__(self, state_size, num_actions):
        super(QNet, self).__init__()
        self.state_size = state_size
        self.num_actions = num_actions
        self.fc1 = nn.Linear(self.state_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, self.num_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CartPole:
    def __init__(self, env):
        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.num_actions = env.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epsilon = 1.0
        self.return_loss = 0.0
        self.memory = []
        self.gamma = 0.95

        self.q_net = QNet(self.state_size, self.num_actions).to(self.device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=0.001)
        self.loss_func = nn.MSELoss()

    def predict_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        rl_debugger.debug(exploration_factor=self.epsilon)
        self.epsilon *= 0.995 + 0.01
        if np.random.random() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            q_values = self.q_net(state_tensor).detach().cpu().numpy()
            action = np.argmax(q_values)
        return action

    def predict_value(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        max_q_value = torch.max(self.q_net(state_tensor)).item()
        return max_q_value

    def train_q_net(self, batch):
        states = torch.tensor([item[0] for item in batch], dtype=torch.float32, device=self.device)
        actions = torch.tensor([item[1] for item in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([item[2] for item in batch], dtype=torch.float32, device=self.device)
        new_states = torch.tensor([item[3] for item in batch], dtype=torch.float32, device=self.device)
        dones = torch.tensor([item[4] for item in batch], dtype=torch.float32, device=self.device)

        # Compute Q values for current states
        q_values = self.q_net(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        # Compute target Q values for next states
        target_q_values = torch.zeros_like(rewards)
        max_q_values = self.q_net(new_states).detach().max(1)[0]
        target_q_values[~dones] = rewards[~dones] + self.gamma * max_q_values[~dones]

        # Compute loss and update Q network
        loss = self.loss_func(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.return_loss = loss.item()

    def get_loss(self):
        return self.return_loss

    def experience_replay(self):
        if len(self.memory) < 33:
            return
        del self.memory[0]


env = gym.make("CartPole-v0")
A2C = CartPole(env)

episodes = 2000
reward_history = []

rl_debugger.set_config(config_path="debugger.yml")
rl_debugger.debug(environment=env, max_steps_per_episode=episodes, max_total_steps=300 * 200)

for i in range(episodes):
    state = env.reset()
    reward_total = 0
    while True:
        state = np.array(state).reshape((1, 4))
        average_best_reward = sum(reward_history[-100:]) / 100.0
        if (average_best_reward) > 195:
            env.render()

        action = A2C.predict_action(state)
        new_state, reward, done, _ = env.step(action)
        reward_total += reward
        A2C.memory.append([state, action, reward, new_state, done])
        A2C.experience_replay()
        state = new_state

        if done:
            if (average_best_reward >= 195):
                print("Finished! Episodes taken: ", i, "average reward: ", average_best_reward)
            print("average reward  = ", average_best_reward, "reward total = ", reward_total, "loss = ", A2C.get_loss())
            reward_history.append(reward_total)
            break
