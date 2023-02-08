import torch
import hive
from hive.runners.utils import load_config
from hive.runners.single_agent_loop import set_up_experiment
from hive.agents.rainbow import RainbowDQNAgent
from hive.replays import PrioritizedReplayBuffer
from debugger import rl_debugger


class DebuggableRainbowAgent(RainbowDQNAgent):

    def update(self, update_info):
        """
        Updates the DQN agent.
        Args:
            update_info: dictionary containing all the necessary information to
            update the agent. Should contain a full transition, with keys for
            "observation", "action", "reward", "next_observation", and "done".
        """
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

            if self._double:
                next_action = self._qnet(*next_state_inputs)
            else:
                next_action = self._target_qnet(*next_state_inputs)
            next_action = next_action.argmax(1)

            if self._distributional:
                current_dist = self._qnet.dist(*current_state_inputs)
                probs = current_dist[torch.arange(actions.size(0)), actions]
                probs = torch.clamp(probs, 1e-6, 1)  # NaN-guard
                log_p = torch.log(probs)
                with torch.no_grad():
                    target_prob = self.target_projection(
                        next_state_inputs, next_action, batch["reward"], batch["done"]
                    )

                loss = -(target_prob * log_p).sum(-1)

            else:
                pred_qvals = pred_qvals[torch.arange(pred_qvals.size(0)), actions]

                next_qvals = self._target_qnet(*next_state_inputs)
                next_qvals = next_qvals[torch.arange(next_qvals.size(0)), next_action]

                q_targets = batch["reward"] + self._discount_rate * next_qvals * (
                        1 - batch["done"]
                )

                rl_debugger.run_debugging(observations=current_state_inputs[0],
                                          model=self._qnet,
                                          predictions=pred_qvals.detach(),
                                          loss_fn=self._loss_fn,
                                          opt=self._optimizer,
                                          actions=actions,
                                          done=update_info["done"]
                                          )

                loss = self._loss_fn(pred_qvals, q_targets)

            if isinstance(self._replay_buffer, PrioritizedReplayBuffer):
                td_errors = loss.sqrt().detach().cpu().numpy()
                self._replay_buffer.update_priorities(batch["indices"], td_errors)
                loss *= batch["weights"]
            loss = loss.mean()

            if self._logger.should_log(self._timescale):
                self._logger.log_scalar(
                    "train_loss",
                    loss,
                    self._timescale,
                )
            loss.backward()
            if self._grad_clip is not None:
                torch.nn.utils.clip_grad_value_(
                    self._qnet.parameters(), self._grad_clip
                )
            self._optimizer.step()


def main():
    hive.registry.register('DebuggableRainbowAgent', DebuggableRainbowAgent, DebuggableRainbowAgent)
    config = load_config(config='agent_configs/custom_rainbow_agent_cartpole.yml')

    rl_debugger.set_config(config_path='debugger.yml')

    runner = set_up_experiment(config)
    runner.run_training()


if __name__ == '__main__':
    main()
