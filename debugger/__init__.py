from debugger.checkers.nn_checkers.on_train_activation_check import OnTrainActivationCheck
from debugger.checkers.nn_checkers.pre_train_bias_check import PreTrainBiasCheck
from debugger.checkers.nn_checkers.pre_train_loss_check import PreTrainLossCheck
from debugger.checkers.nn_checkers.pre_train_observations_check import PreTrainObservationsCheck
from debugger.checkers.nn_checkers.on_train_bias_check import OnTrainBiasCheck
from debugger.checkers.nn_checkers.on_train__loss_check import OnTrainLossCheck
from debugger.checkers.nn_checkers.on_train_weight_check import OnTrainWeightsCheck
from debugger.checkers.nn_checkers.pre_train_proper_fitting_check import PreTrainProperFittingCheck
from debugger.checkers.nn_checkers.pre_train_gradient_check import PreTrainGradientCheck
from debugger.checkers.nn_checkers.pre_train_weights_check import PreTrainWeightsCheck
from debugger.debugger_interface import DebuggerInterface
from debugger.debugger_factory import DebuggerFactory
from debugger.utils.registry import registry

registry.register("PreTrainObservation", PreTrainObservationsCheck, PreTrainObservationsCheck)
registry.register("PreTrainWeight", PreTrainWeightsCheck, PreTrainWeightsCheck)
registry.register("PreTrainBias", PreTrainBiasCheck, PreTrainBiasCheck)
registry.register("PreTrainLoss", PreTrainLossCheck, PreTrainLossCheck)
registry.register("PreTrainProperFitting", PreTrainProperFittingCheck, PreTrainProperFittingCheck)
registry.register("OnTrainActivation", OnTrainActivationCheck, OnTrainActivationCheck)
registry.register("PreTrainGradient", PreTrainGradientCheck, PreTrainGradientCheck)
registry.register("OnTrainBias", OnTrainBiasCheck, OnTrainBiasCheck)
registry.register("OnTrainWeight", OnTrainWeightsCheck, OnTrainWeightsCheck)
registry.register("OnTrainLoss", OnTrainLossCheck, OnTrainLossCheck)

get_debugger = getattr(registry, f"get_{DebuggerInterface.type_name()}")


class DebuggerSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = DebuggerFactory()
        return cls._instance


# Create an instance of the singleton class
rl_debugger = DebuggerSingleton()

