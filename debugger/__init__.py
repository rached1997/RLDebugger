from debugger.checkers.nn_checkers.on_train_activation_check import OnTrainActivationCheck
from debugger.checkers.nn_checkers.pre_train_bias_check import PreTrainBiasCheck
from debugger.checkers.nn_checkers.pre_train_loss_check import PreTrainLossCheck
from debugger.checkers.nn_checkers.on_train_bias_check import OnTrainBiasCheck
from debugger.checkers.nn_checkers.on_train_loss_check import OnTrainLossCheck
from debugger.checkers.nn_checkers.on_train_weight_check import OnTrainWeightsCheck
from debugger.checkers.nn_checkers.pre_train_proper_fitting_check import PreTrainProperFittingCheck
from debugger.checkers.nn_checkers.pre_train_gradient_check import PreTrainGradientCheck
from debugger.checkers.nn_checkers.pre_train_weights_check import PreTrainWeightsCheck
from debugger.checkers.rl_checkers.on_train_action_check import OnTrainActionCheck
from debugger.checkers.rl_checkers.on_train_exploration_param_check import OnTrainExplorationParameterCheck
from debugger.checkers.rl_checkers.on_train_uncertainty_action_check import OnTrainUncertaintyActionCheck
from debugger.checkers.rl_checkers.on_train_agent_check import OnTrainAgentCheck
from debugger.checkers.rl_checkers.on_train_reward_check import OnTrainRewardsCheck
from debugger.checkers.rl_checkers.on_train_states_check import OnTrainStatesCheck
from debugger.checkers.rl_checkers.on_train_value_function_check import OnTrainValueFunctionCheck
from debugger.checkers.rl_checkers.pre_train_environment_check import PreTrainEnvironmentCheck
from debugger.debugger_interface import DebuggerInterface
from debugger.debugger_factory import DebuggerFactory
from debugger.utils.registry import registry

registry.register("PreTrainWeight", PreTrainWeightsCheck, PreTrainWeightsCheck)
registry.register("PreTrainBias", PreTrainBiasCheck, PreTrainBiasCheck)
registry.register("PreTrainLoss", PreTrainLossCheck, PreTrainLossCheck)
registry.register("PreTrainProperFitting", PreTrainProperFittingCheck, PreTrainProperFittingCheck)
registry.register("OnTrainActivation", OnTrainActivationCheck, OnTrainActivationCheck)
registry.register("PreTrainGradient", PreTrainGradientCheck, PreTrainGradientCheck)
registry.register("OnTrainBias", OnTrainBiasCheck, OnTrainBiasCheck)
registry.register("OnTrainWeight", OnTrainWeightsCheck, OnTrainWeightsCheck)
registry.register("OnTrainLoss", OnTrainLossCheck, OnTrainLossCheck)
registry.register("PreTrainEnvironment", PreTrainEnvironmentCheck, PreTrainEnvironmentCheck)
registry.register("OnTrainState", OnTrainStatesCheck, OnTrainStatesCheck)
registry.register("OnTrainReward", OnTrainRewardsCheck, OnTrainRewardsCheck)
registry.register("OnTrainAgent", OnTrainAgentCheck, OnTrainAgentCheck)
registry.register("OnTrainValueFunction", OnTrainValueFunctionCheck, OnTrainValueFunctionCheck)

registry.register("Action", OnTrainActionCheck, OnTrainActionCheck)
registry.register("UncertaintyAction", OnTrainUncertaintyActionCheck, OnTrainUncertaintyActionCheck)
registry.register("OnTrainExplorationParameter", OnTrainExplorationParameterCheck, OnTrainExplorationParameterCheck)

get_debugger = getattr(registry, f"get_{DebuggerInterface.type_name()}")


class DebuggerSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = DebuggerFactory()
        return cls._instance


# Create an instance of the singleton class
rl_debugger = DebuggerSingleton()
