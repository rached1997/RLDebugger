import torch
import numpy as np
import hive
from hive.runners.utils import load_config
from hive.runners.single_agent_loop import set_up_experiment
from hive.agents.dqn import DQNAgent
from debugger import rl_debugger


# uncomment the following line if you want to use the custom check provided in the examples folder
# from custom_checker import CustomChecker


class DebuggableDQNAgent(DQNAgent):
    @torch.no_grad()
    def act(self, observation):
        if self._training:
            if not self._learn_schedule.get_value():
                epsilon = 1.0
            else:
                epsilon = self._epsilon_schedule.update()
            if self._logger.update_step(self._timescale):
                self._logger.log_scalar("epsilon", epsilon, self._timescale)
        else:
            epsilon = self._test_epsilon
        observation = torch.tensor(
            np.expand_dims(observation, axis=0), device=self._device
        ).float()
        qvals = self._qnet(observation)
        if self._rng.random() < epsilon:
            action = self._rng.integers(self._act_dim)
        else:
            # Note: not explicitly handling the ties
            action = torch.argmax(qvals).item()

        if (
                self._training
                and self._logger.should_log(self._timescale)
                and self._state["episode_start"]
        ):
            self._state["episode_start"] = False

        rl_debugger.debug(actions_probs=qvals.detach(), exploration_factor=epsilon, model=self._qnet)
        return action

    def update(self, update_info):
        if update_info["done"]:
            self._state["episode_start"] = True

        if not self._training:
            return

        self._replay_buffer.add(**self.preprocess_update_info(update_info))
        if (
                self._learn_schedule.update()
                and self._replay_buffer.size() > 0
                and self._update_period_schedule.update()
        ):
            batch = self._replay_buffer.sample(batch_size=self._batch_size)
            (
                current_state_inputs,
                next_state_inputs,
                batch,
            ) = self.preprocess_update_batch(batch)

            # Compute predicted Q values
            self._optimizer.zero_grad()
            pred_qvals = self._qnet(*current_state_inputs)
            actions = batch["action"].long()
            pred_qvals = pred_qvals[torch.arange(pred_qvals.size(0)), actions]

            # Compute 1-step Q targets
            next_qvals = self._target_qnet(*next_state_inputs)
            next_qvals, _ = torch.max(next_qvals, dim=1)
            q_targets = batch["reward"] + self._discount_rate * next_qvals * (
                    1 - batch["done"]
            )

            rl_debugger.debug(
                target_model=self._target_qnet,
                target_model_update_period=self._target_net_update_schedule._total_period,
                target_net_update_fraction=1,
                loss_fn=self._loss_fn,
                opt=self._optimizer,
                targets=q_targets.detach(),
                training_observations=current_state_inputs[0],
                actions=actions,
                discount_rate=self._discount_rate,
                predicted_next_vals=next_qvals.detach(),
                steps_rewards=batch["reward"],
                steps_done=batch["done"]
            )

            loss = self._loss_fn(pred_qvals, q_targets).mean()

            loss.backward()
            if self._grad_clip is not None:
                torch.nn.utils.clip_grad_value_(
                    self._qnet.parameters(), self._grad_clip
                )
            self._optimizer.step()

        # Update target network
        if self._target_net_update_schedule.update():
            self._update_target()
