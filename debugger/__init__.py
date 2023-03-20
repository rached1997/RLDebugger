from debugger.checkers.nn_checkers.activation_check import ActivationCheck
from debugger.checkers.nn_checkers.bias_check import BiasCheck
from debugger.checkers.nn_checkers.loss_check import LossCheck
from debugger.checkers.nn_checkers.weight_check import WeightsCheck
from debugger.checkers.nn_checkers.proper_fitting_check import ProperFittingCheck
from debugger.checkers.nn_checkers.gradient_check import GradientCheck
from debugger.checkers.rl_checkers.action_check import ActionCheck
from debugger.checkers.rl_checkers.exploration_param_check import (
    ExplorationParameterCheck,
)
from debugger.checkers.rl_checkers.steps_check import StepCheck
from debugger.checkers.rl_checkers.uncertainty_action_check import (
    UncertaintyActionCheck,
)
from debugger.checkers.rl_checkers.agent_check import AgentCheck
from debugger.checkers.rl_checkers.reward_check import RewardsCheck
from debugger.checkers.rl_checkers.states_check import StatesCheck
from debugger.checkers.rl_checkers.q_target_check import QTargetCheck
from debugger.checkers.rl_checkers.environment_check import EnvironmentCheck
from debugger.debugger_interface import DebuggerInterface
from debugger.debugger_factory import DebuggerFactory
from debugger.utils.registry import registry

registry.register("ProperFitting", ProperFittingCheck, ProperFittingCheck)
registry.register("Activation", ActivationCheck, ActivationCheck)
registry.register("Gradient", GradientCheck, GradientCheck)
registry.register("Bias", BiasCheck, BiasCheck)
registry.register("Weight", WeightsCheck, WeightsCheck)
registry.register("Loss", LossCheck, LossCheck)

registry.register("Environment", EnvironmentCheck, EnvironmentCheck)
registry.register("State", StatesCheck, StatesCheck)
registry.register("Reward", RewardsCheck, RewardsCheck)
registry.register("Agent", AgentCheck, AgentCheck)
registry.register("QTarget", QTargetCheck, QTargetCheck)
registry.register("Action", ActionCheck, ActionCheck)
registry.register("UncertaintyAction", UncertaintyActionCheck, UncertaintyActionCheck)
registry.register(
    "ExplorationParameter", ExplorationParameterCheck, ExplorationParameterCheck
)
registry.register("Step", StepCheck, StepCheck)

get_debugger = getattr(registry, f"get_{DebuggerInterface.type_name()}")


class DebuggerSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = DebuggerFactory()
            cls._instance.set_config()
        return cls._instance


# Create an instance of the singleton class
rl_debugger = DebuggerSingleton()
