import torch
import numpy as np
import hive
from hive.runners.utils import load_config
from hive.runners.single_agent_loop import set_up_experiment
from hive.agents.dqn import DQNAgent
from debugger import rl_debugger


# uncomment the following line if you want to use the custom check provided in the example folder
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
            # rl_debugger.run_debugging(actions_probs=torch.softmax(torch.randn(1, self._act_dim), dim=1).to('cuda'))
        else:
            # Note: not explicitly handling the ties
            action = torch.argmax(qvals).item()
            # rl_debugger.run_debugging(actions_probs=qvals)

        if (
                self._training
                and self._logger.should_log(self._timescale)
                and self._state["episode_start"]
        ):
            self._state["episode_start"] = False
        if self._training:
            rl_debugger.run_debugging(actions_probs=qvals)
        # rl_debugger.run_debugging(model=self._qnet)
        # rl_debugger.run_debugging(model=self._qnet, observations=observation, exploration_factor=self._test_epsilon)
        # rl_debugger.run_debugging(environment=environment._env, model=self._qnet, exploration_factor=epsilon)
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
            q_targets = batch["reward"] + self._discount_rate * next_qvals * (1 - batch["done"])

            # rl_debugger.run_debugging(model=self._qnet,
            #                           loss_fn=self._loss_fn,
            #                           opt=self._optimizer,
            #                           target_model=self._target_qnet,
            #                           target_model_update_period=self._target_net_update_schedule._total_period,
            #                           target_net_update_fraction=1,
            #                           targets=q_targets.detach(),
            #                           # predictions=pred_qvals.detach(),
            #                           actions=actions,
            #                           # predicted_next_vals=next_qvals.detach(),
            #                           discount_rate=self._discount_rate,
            #                           )

            loss = self._loss_fn(pred_qvals, q_targets).mean()

            loss.backward()
            if self._grad_clip is not None:
                torch.nn.utils.clip_grad_value_(
                    self._qnet.parameters(), self._grad_clip
                )
            self._optimizer.step()

        # Update target network
        self._target_qnet = self._qnet
        if self._target_net_update_schedule.update():
            self._update_target()


# def main():
#     hive.registry.register('DebuggableDQNAgent', DebuggableDQNAgent, DebuggableDQNAgent)
#     config = load_config(config='agent_configs/custom_agent_cartpole.yml')
#
#     # uncomment the following line if you want to use the custom checker
#     # the register method should be called before the set_config method
#     # rl_debugger.register(checker_name="CustomChecker", checker_class=CustomChecker)
#     rl_debugger.set_config(config_path="debugger.yml")
#
#     runner = set_up_experiment(config)
#     runner.run_training()
#
#
# if __name__ == '__main__':
#     main()
