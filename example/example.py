import torch
import copy
import hive
from hive.runners.utils import load_config
from hive.runners.single_agent_loop import set_up_experiment
from hive.agents.dqn import DQNAgent
from debugger import rl_debugger


class DebuggableDQNAgent(DQNAgent):

    def update(self, update_info):
        if update_info["done"]:
            self._state["episode_start"] = True

        if not self._training:
            return

        # Add the most recent transition to the replay buffer.
        self._replay_buffer.add(**self.preprocess_update_info(update_info))

        # Update the q network based on a sample batch from the replay buffer.
        # If the replay buffer doesn't have enough samples, catch the exception
        # and move on.
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

            # TODO: change copy.deepcopy please
            rl_debugger.run_debugging(observations=copy.deepcopy(current_state_inputs[0].numpy()),
                                      model=copy.deepcopy(self._qnet),
                                      labels=copy.deepcopy(q_targets),
                                      predictions=copy.deepcopy(pred_qvals.detach()),
                                      loss=copy.deepcopy(self._loss_fn),
                                      opt=copy.deepcopy(self._optimizer),
                                      actions=copy.deepcopy(actions),
                                      done=copy.deepcopy(update_info["done"])
                                      )

            loss = self._loss_fn(pred_qvals, q_targets).mean()

            if self._logger.should_log(self._timescale):
                self._logger.log_scalar("train_loss", loss, self._timescale)

            loss.backward()
            if self._grad_clip is not None:
                torch.nn.utils.clip_grad_value_(
                    self._qnet.parameters(), self._grad_clip
                )
            self._optimizer.step()

        # Update target network
        if self._target_net_update_schedule.update():
            self._update_target()


hive.registry.register('DebuggableDQNAgent', DebuggableDQNAgent, DebuggableDQNAgent)

config = load_config(config='dqn_VR.yml')

x = rl_debugger

rl_debugger.set_config(config["debugger"]["kwargs"]["check_type"])

runner = set_up_experiment(config)
runner.run_training()
