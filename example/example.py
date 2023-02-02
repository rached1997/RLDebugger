import torch
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

            rl_debugger.run_debugging(observations=current_state_inputs[0],
                                      model=self._qnet,
                                      labels=q_targets,
                                      predictions=pred_qvals.detach(),
                                      loss_fn=self._loss_fn,
                                      opt=self._optimizer,
                                      actions=actions,
                                      done=update_info["done"]
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


def main():
    hive.registry.register('DebuggableDQNAgent', DebuggableDQNAgent, DebuggableDQNAgent)
    config = load_config(config='custom_agent.yml')

    rl_debugger.set_config(config_path='debugger.yml')

    runner = set_up_experiment(config)
    runner.run_training()


if __name__ == '__main__':
    main()
