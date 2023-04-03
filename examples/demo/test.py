import gym
import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from debugger import rl_debugger

# hyper parameters
EPISODES = 200  # number of episodes
EPS_START = 0.9  # e-greedy threshold start value
EPS_END = 0.05  # e-greedy threshold end value
EPS_DECAY = 200  # e-greedy threshold decay
GAMMA = 0.8  # Q-learning discount factor
LR = 0.001  # NN optimizer learning rate
HIDDEN_LAYER = 256  # NN hidden layer size
BATCH_SIZE = 64  # Q-learning batch size


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Network(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, HIDDEN_LAYER)
        self.l2 = nn.Linear(HIDDEN_LAYER, 2)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


rl_debugger.set_config(config_path="demo.yml")
env = gym.make('CartPole-v0')
rl_debugger.debug(
        environment=env,
        max_reward=env.spec.reward_threshold,
        max_steps_per_episode=env.spec.reward_threshold,
        max_total_steps=1000)
model = Network()
memory = ReplayMemory(10000)
optimizer = optim.Adam(model.parameters(), LR)
steps_done = 0
episode_durations = []


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    rl_debugger.debug(model=model)
    if sample > eps_threshold:
        return model(Variable(state, volatile=True).type(torch.FloatTensor)).data.max(1)[1].view(1, 1)
    else:
        return torch.LongTensor([[random.randrange(2)]])


def run_episode(e, environment):
    state = environment.reset()
    steps = 0
    while True:
        action = select_action(torch.FloatTensor([state]))
        next_state, reward, done, _ = environment.step(np.array(action)[0, 0])

        # negative reward when attempt ends
        if done:
            reward = -1

        memory.push((torch.FloatTensor([state]),
                     action,  # action is already a tensor
                     torch.FloatTensor([next_state]),
                     torch.FloatTensor([reward])))

        learn()

        state = next_state
        steps += 1

        if done:
            print("{2} Episode {0} finished after {1} steps"
                  .format(e, steps, '\033[92m' if steps >= 195 else '\033[99m'))
            break


def learn():
    if len(memory) < BATCH_SIZE:
        return

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(BATCH_SIZE)
    batch_state, batch_action, batch_next_state, batch_reward = zip(*transitions)

    batch_state = Variable(torch.cat(batch_state))
    batch_action = Variable(torch.cat(batch_action))
    batch_reward = Variable(torch.cat(batch_reward))
    batch_next_state = Variable(torch.cat(batch_next_state))

    # current Q values are estimated by NN for all actions
    current_q_values = model(batch_state).gather(1, batch_action)
    # expected Q values are estimated from actions which gives maximum Q value
    max_next_q_values = model(batch_next_state).detach().max(1)[0]
    expected_q_values = batch_reward + (GAMMA * max_next_q_values)

    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(current_q_values, expected_q_values)

    # backpropagation of loss to NN
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


for e in range(EPISODES):
    run_episode(e, env)
env.close()

print(rl_debugger.step_num)

from debugger import reset_debugger

reset_debugger()

from debugger import rl_debugger

print(rl_debugger.step_num)