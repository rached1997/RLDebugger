from debugger.checkers.nn_checkers.activation_check import ActivationCheck
from debugger.checkers.nn_checkers.bias_check import BiasCheck
from debugger.checkers.nn_checkers.loss_check import LossCheck
from debugger.checkers.nn_checkers.observations_check import ObservationsCheck
from debugger.checkers.nn_checkers.overfit_bias_check import OverfitBiasCheck
from debugger.checkers.nn_checkers.overfit_loss_check import OverfitLossCheck
from debugger.checkers.nn_checkers.overfit_weight_check import OverfitWeightsCheck
from debugger.checkers.nn_checkers.proper_fitting_check import ProperFittingCheck
from debugger.checkers.nn_checkers.gradient_check import GradientCheck
from debugger.checkers.nn_checkers.weights_check import WeightsCheck
from debugger.debugger_interface import DebuggerInterface
from debugger.utils.registry import registry
# Todo
#  registry.register_all(
#     Debugger,
#     {
#         "NullDebugger": NullDebugger,
#         "PreCheckDebugger": PreCheckDebugger,
#         "PostCheckDebugger": PostCheckDebugger,
#         "OnTrainingCheckDebugger": OnTrainingCheckDebugger,
#         "CompositeDebugger": CompositeDebugger,
#     },
# )

registry.register("Observation", ObservationsCheck, ObservationsCheck)
registry.register("Weight", WeightsCheck, WeightsCheck)
registry.register("Bias", BiasCheck, BiasCheck)
registry.register("Loss", LossCheck, LossCheck)
registry.register("ProperFitting", ProperFittingCheck, ProperFittingCheck)
registry.register("Activation", ActivationCheck, ActivationCheck)
registry.register("Gradient", GradientCheck, GradientCheck)
registry.register("OverfitBias", OverfitBiasCheck, OverfitBiasCheck)
registry.register("OverfitWeight", OverfitWeightsCheck, OverfitWeightsCheck)
registry.register("OverfitLoss", OverfitLossCheck, OverfitLossCheck)


get_debugger = getattr(registry, f"get_{DebuggerInterface.type_name()}")
