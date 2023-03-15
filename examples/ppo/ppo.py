# Based on implementation from Barhate, Nikhil repository (https://github.com/nikhilbarhate99/PPO-PyTorch)
# Accessed on 3/11/2023
import torch
import torch.nn as nn
from examples.ppo.actor_critic import RolloutBuffer, ActorCritic
from debugger import rl_debugger
import numpy as np

# Set seeds for Torch
torch.manual_seed(42)

# Set seeds for Torch's backend (cuda or GPU)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(42)
# else:
#     np.random.seed(42)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


class PPO:
    def __init__(
        self, state_dim, action_dim, lr_actor, lr_critic, gamma, k_epochs, eps_clip
    ):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = k_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim).to(get_device())
        self.optimizer = torch.optim.Adam(
            [
                {"observed_params": self.policy.actor.parameters(), "lr": lr_actor},
                {"observed_params": self.policy.critic.parameters(), "lr": lr_critic},
            ]
        )

        self.policy_old = ActorCritic(state_dim, action_dim).to(get_device())
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, incr):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(get_device())
            action, action_logprob, state_val, action_probs = self.policy_old.act(state)

        # if incr % 2:
        rl_debugger.debug(
            model=self.policy_old.actor, actions=action, actions_probs=action_probs
        )

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return action.item()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(
            reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(get_device())
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = (
            torch.squeeze(torch.stack(self.buffer.states, dim=0))
            .detach()
            .to(get_device())
        )
        old_actions = (
            torch.squeeze(torch.stack(self.buffer.actions, dim=0))
            .detach()
            .to(get_device())
        )
        old_logprobs = (
            torch.squeeze(torch.stack(self.buffer.logprobs, dim=0))
            .detach()
            .to(get_device())
        )
        old_state_values = (
            torch.squeeze(torch.stack(self.buffer.state_values, dim=0))
            .detach()
            .to(get_device())
        )
        rl_debugger.debug(training_observations=old_states)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions
            )

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )

            # final loss of clipped objective PPO
            loss = (
                -torch.min(surr1, surr2)
                + 0.5 * self.MseLoss(state_values, rewards)
                - 0.01 * dist_entropy
            )

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

        return old_states
