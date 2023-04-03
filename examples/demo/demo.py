import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

# hyper parameters
EPISODES = 200  # number of episodes
GAMMA = 0.8  # Q-learning discount factor
LR = 0.001  # NN optimizer learning rate
HIDDEN_LAYER = 256  # NN hidden layer size
BATCH_SIZE = 64  # Q-learning batch size
MAX_TOTAL_STEPS = 2000  # number of all steps in the experiment
MAX_STEPS_PER_EPS = 475  # max number steps in one episode


class Network(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(4, HIDDEN_LAYER)
        self.l2 = nn.Linear(HIDDEN_LAYER, 2)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


def run_one_episode(episode, environment):
    state = environment.reset()
    steps = 0
    while True:
        action = model(Variable(torch.FloatTensor([state]),
                                volatile=True).type(torch.FloatTensor)).data.max(1)[1].view(1, 1)

        next_state, reward, done, _ = environment.step(np.array(action)[0, 0])
        learn()
        state = next_state
        steps += 1
        if done:
            print("{2} Episode {0} finished after {1} steps"
                  .format(episode, steps, '\033[92m' if steps >= 195 else '\033[99m'))
            break


def learn():
    batch_state = torch.rand(size=(BATCH_SIZE, 4), dtype=torch.float32)
    batch_action = torch.randint(high=2, size=(64, 1))
    batch_reward = torch.randint(high=EPISODES, size=(64,)).type(torch.FloatTensor)
    batch_next_state = torch.rand(size=(BATCH_SIZE, 4), dtype=torch.float32)

    current_q_values = model(batch_state).gather(1, batch_action)
    max_next_q_values = model(batch_next_state).detach().max(1)[0]
    expected_q_values = batch_reward + (GAMMA * max_next_q_values)

    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(current_q_values, expected_q_values)
    # backpropagation of loss to NN
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


env = gym.make('CartPole-v1')
MAX_REWARD = env.spec.reward_threshold
model = Network()
optimizer = optim.Adam(model.parameters(), LR)

for e in range(EPISODES):
    run_one_episode(e, env)
env.close()
