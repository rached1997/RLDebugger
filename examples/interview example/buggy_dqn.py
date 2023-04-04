import gym
import math
import random
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from debugger import rl_debugger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, state, action, next_state, reward):
        """Save a transition"""
        self.memory.append(Transition(state, action, next_state, reward))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(
        BATCH_SIZE)  # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    batch = Transition(*zip(
        *transitions))  # detailed explanation). This converts batch-array of Transitions to Transition of batch-arrays.

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch  # Compute the expected Q values

    criterion = nn.SmoothL1Loss()
    rl_debugger.debug(model=policy_net, actions= action_batch, actions_probs=policy_net(state_batch).detach())

    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)  # In-place gradient clipping
    optimizer.step()


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


env = gym.make("CartPole-v1")

BATCH_SIZE = 128  # BATCH_SIZE is the number of transitions sampled from the replay buffer
GAMMA = 0.99  # GAMMA is the discount factor as mentioned in the previous section
EPS_START = 0.9  # EPS_START is the starting value of epsilon
EPS_END = 0.05  # EPS_END is the final value of epsilon
EPS_DECAY = 1000  # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
TAU = 0.005  # TAU is the update rate of the target network
LR = 1e-4  # LR is the learning rate of the AdamW optimizer

n_actions = env.action_space.n  # Get number of actions from gym action space
state = env.reset()
n_observations = len(state)  # Get the number of state observations

policy_net = DQN(n_observations, n_actions).to(device)

target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0
num_episodes = 400
max_step_per_episode = 500

rl_debugger.set_config(config_path="debugger.yml")
rl_debugger.debug(environment=env, max_steps_per_episode=max_step_per_episode,
                  max_total_steps=num_episodes * max_step_per_episode,
                  max_reward=env.spec.reward_threshold)

for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in range(max_step_per_episode):
        action = select_action(state)
        observation, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        if done:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            print("Episode ", i_episode, " Reward ", t)
            break
