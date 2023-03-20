import copy
import re
import torch.nn

from debugger.config_data_classes.nn_checkers.activation_config import ActivationConfig
from debugger.debugger_interface import DebuggerInterface
from debugger.utils.utils import (
    get_activation_max_min_bound,
    transform_2d,
    compute_ro_B,
    pure_f_test,
)
from debugger.utils.model_params_getters import (
    is_activation_function,
    get_last_layer_activation,
)
import numpy as np
import torch


class ActivationCheck(DebuggerInterface):
    """
    The check is in charge of verifying the activation functions during training.
    For more details on the specific checks performed, refer to the `run()` function.
    """

    def __init__(self):
        """
        Initializes the following parameters:
            * outputs_metadata: a dictionary tracking statistical features in the activations
        """
        super().__init__(check_type="Activation", config=ActivationConfig)
        self._acts_data = {}
        self._outputs_metadata = {
            "non_zero_variance": {"status": None},
            "max_abs_greater_than_one": {"status": None},
            "can_be_negative": {"status": None},
        }

    def run(self, training_observations: torch.Tensor, model: torch.nn.Module) -> None:
        """
        -----------------------------------   I. Introduction of the Activation Check  --------------------------------

        This class performs multiple checks on the activation values of the neural network. Activation consists of
        non-linear functions applied to each output neuron in order to determine if the neuron will be activated or
        not. The activations are the ones that let the neural network learn nonlinear and complex patterns in the
        data. This class checks if the activations are behaving normally and don't show any unusual behavior.

        ------------------------------------------   II. The performed checks  -----------------------------------------

        This class does multiple checks on the activation values during the training. The checks it does are :
            (1) Checks on the activations of the output layer:
                a. Checks the numerical Stability of the output activation by verifying if there are any NaN or
                infinite values.
                b. Checks that the activations are constantly changing.
                c. Checks if the Outputs represents probabilities (this only works when the output layer is a softmax
                   layer).

            Checks on each activation layer :
                (2) Checks if the activations produced by the specified activation layer are within the expected
                range of values based on the activation function used.
                (3) Validates the numerical stability of activation values in the given activation layer
                (4) Detects saturation in the activation values for bounded activation functions.
                (5) Detects dead neurons in the activation layer
                (6) Checks the stability of the activation values over multiple iterations.

        ------------------------------------   III. The potential Root Causes  -----------------------------------------

        The potential root causes behind the warnings that can be detected are
            - Bad neural network hyperparameters (checks triggered : 1..6)
            - Bad architecture of the neural network (checks triggered : 1..6)
            - Poor initialisation of the weights and biases (checks triggered : 1..6)
            - Using the wrong activation function (checks triggered : 1..6)
            - Coding error (checks triggered : 1..6)
            - Using a bad output layer (checks triggered : 1)

        --------------------------------------   IV. The Recommended Fixes  --------------------------------------------

        The recommended fixes for the detected issues:
            - Change the neural network's hyperparameters to stabilize more the learning ( checks tha can be fixed: 1..6)
            - Change the number of layers or parameters of our neural network ( checks tha can be fixed: 1..6)
            - Verify that you are using the activation function correctly and in the right application ( checks tha can be fixed: 1..6)
            - Verify that you are using the right output function correctly  ( checks tha can be fixed: 1..6)

        Examples
        --------
        To perform activation checks, the debugger needs to be called when updating the agent. Note that
        the debugger needs to be called before performing the backward prop (.backward()).

        >>> from debugger import rl_debugger
        >>> ...
        >>> batch = replay_buffer.sample(batch_size=32)
        >>> rl_debugger.debug(model=qnet, training_observations=batch["state"])
        >>> loss = loss_fn(pred_qvals, q_targets).mean()
        >>> loss.backward()

        Args:
            training_observations (Tensor): A sample of observations collected during the training
            model (torch.nn.Module)): The model being trained
        """
        if self.config.disabled:
            return
        model = copy.deepcopy(model)
        activations = {}

        def hook(module, input, output):
            activations[module] = output

        def get_activation():
            for name, layer in model.named_modules():
                if is_activation_function(layer):
                    layer.register_forward_hook(hook)

        get_activation()
        outputs = model(training_observations)
        if self._acts_data == {}:
            self.set_acts_data(training_observations.shape[0], activations)

        self.update_outs_conds(outputs)

        if self.check_period():
            self.check_outputs(outputs, get_last_layer_activation(model))

        for i, (acts_name, acts_array) in enumerate(activations.items()):
            acts_name = re.sub(r"\([^()]*\)", "", str(acts_name)) + "_" + str(i)
            acts_buffer = self.update_buffer(acts_name, acts_array)
            if self.iter_num >= self.config.start and self.check_period():
                self.check_activations_range(acts_name, acts_buffer)
                if self.check_numerical_instabilities(acts_name, acts_array):
                    continue
                if acts_name in ["Sigmoid", "Tanh"]:
                    self.check_saturated_layers(acts_name, acts_buffer)
                else:
                    self.check_dead_layers(acts_name, acts_buffer)

                self.check_acts_distribution(acts_name, acts_buffer)
                self.config.disabled = True

    def update_outs_conds(self, outs_array: torch.Tensor) -> None:
        """
        updates the metadata needed in the activation functions checks.
        Args:
            outs_array (Tensor): The activations of the output layer.
        Returns:
            None
        """
        if self._outputs_metadata["non_zero_variance"]["status"] is None:
            self._outputs_metadata["non_zero_variance"]["status"] = (
                outs_array.var(dim=0) > 0
            )
            self._outputs_metadata["max_abs_greater_than_one"]["status"] = (
                torch.abs(outs_array) > 1
            ).any(dim=0)
            self._outputs_metadata["can_be_negative"]["status"] = (outs_array < 0).any(
                dim=0
            )
        else:
            self._outputs_metadata["non_zero_variance"]["status"] |= (
                outs_array.var(dim=0) > 0
            )
            self._outputs_metadata["max_abs_greater_than_one"]["status"] |= (
                torch.abs(outs_array) > 1
            ).any(dim=0)
            self._outputs_metadata["can_be_negative"]["status"] |= (outs_array < 0).any(
                dim=0
            )

    def check_outputs(
        self, outs_array: torch.Tensor, last_layer_activation_name: str
    ) -> None:
        """
        Validate the Output Activation Domain of the model. This function performs the following checks:
        1. Numerical Stability of the output activation by verifying if there are any NaN or infinite values.
        2. Variance of the output layer's activations to ensure that they are constantly changing.
        3. Outputs are probabilities, i.e., positive values within [0, 1] and summing to one for multidimensional
        outputs.
        Args:
            last_layer_activation_name: last layer activation name.
            outs_array (Tensor): The activations of the output layer.
        Returns:
            None
        """
        if torch.isinf(outs_array).any():
            self.error_msg.append(self.main_msgs["out_inf"])
            return
        elif torch.isnan(outs_array).any():
            self.error_msg.append(self.main_msgs["out_nan"])
            return
        if (self._outputs_metadata["non_zero_variance"]["status"] == False).any():
            self.config.patience -= 1
            if self.config.patience <= 0:
                self.error_msg.append(self.main_msgs["out_cons"])
        else:
            self.config.patience = self.config.Output.patience

        if "Softmax" == last_layer_activation_name:
            if outs_array.shape[1] == 1:
                positive = (outs_array >= 0.0).all() and (outs_array <= 1.0).all()
                if not positive:
                    self.error_msg.append(self.main_msgs["output_invalid"])
            else:
                # cannot check sum to 1.0 because of https://randomascii.wordpress.com/2012/02/25/comparing-floating
                # -point-numbers-2012-edition/
                sum_to_one = (torch.sum(outs_array, dim=1) > 0.95).all() and (
                    torch.sum(outs_array, dim=1) < 1.05
                ).all()
                positive = (outs_array >= 0.0).all()
                valid_n_outs = outs_array.shape[1]
                if not (positive and sum_to_one and valid_n_outs):
                    self.error_msg.append(self.main_msgs["output_invalid"])

    def check_activations_range(self, acts_name: str, acts_array: np.ndarray) -> None:
        """
        Checks if the activations produced by the specified activation layer are within the expected range
        of values based on the activation function used.
        Args:
            acts_name (str): The name of the activation layer to be checked. The name should contain the name of the
            activation function used, e.g. "Relu", "Tanh", "Sigmoid", etc.
            acts_array (np.ndarray): The activations obtained from the specified activation layer.
        Returns:
            None
        """
        if self.config.Range.disabled:
            return
        acts_max_bound, acts_min_bound = get_activation_max_min_bound(str(acts_name))
        if (acts_array < acts_min_bound).any():
            main_msg = self.main_msgs["act_ltn"].format(acts_name, acts_min_bound)
            self.error_msg.append(main_msg)
        if (acts_array > acts_max_bound).any():
            main_msg = self.main_msgs["act_gtn"].format(acts_name, acts_max_bound)
            self.error_msg.append(main_msg)

    def check_numerical_instabilities(
        self, acts_name: str, acts_array: torch.Tensor
    ) -> bool:
        """
        Validates the numerical stability of activation values in the given activation layer.
        Args:
            acts_name (str): The name of the activation layer to be validated. The name should include the name of the
            activation function used, such as "Relu", "Tanh", "Sigmoid", etc.
            acts_array (Tensor): The activations obtained from the specified activation layer.
        Returns:
            (bool): True if there is any NaN or infinite value present, False otherwise.
        """

        if self.config.Numerical_Instability.disabled:
            return False
        if torch.isinf(acts_array).any():
            self.error_msg.append(self.main_msgs["act_inf"].format(acts_name))
            return True
        if torch.isnan(acts_array).any():
            self.error_msg.append(self.main_msgs["act_nan"].format(acts_name))
            return True
        return False

    def check_saturated_layers(self, acts_name: str, acts_array: np.ndarray) -> None:
        """
        Detects saturation in the activation values for bounded activation functions using the 𝜌𝐵 metric as proposed in
        the paper "Measuring Saturation in Neural Networks" by Rakitianskaia and Engelbrecht.
            - link: https://ieeexplore.ieee.org/document/7376778
        Args:
            acts_name (str): The name of the activation layer to be validated. The name should include the name of the
            activation function used, such as "Relu", "Tanh", "Sigmoid", etc.
            acts_array (np.ndarray): The activations obtained from the specified activation layer.
        Returns:
            None
        """
        if self.config.Saturation.disabled:
            return
        acts_array = transform_2d(acts_array, keep="last").numpy()
        ro_Bs = np.apply_along_axis(
            compute_ro_B,
            0,
            acts_array,
            min_out=self.config.Saturation.ro_histo_min,
            max_out=self.config.Saturation.ro_histo_max,
            bins_count=self.config.Saturation.ro_histo_bins_count,
        )
        saturated_count = np.count_nonzero(ro_Bs > self.config.Saturation.ro_max_thresh)
        saturated_ratio = saturated_count / ro_Bs.shape[0]
        if saturated_ratio > self.config.sat.neurons_ratio_max_thresh:
            main_msg = self.main_msgs["act_sat"]
            self.error_msg.append(
                main_msg.format(saturated_count, ro_Bs.size, acts_name)
            )

    def check_dead_layers(self, acts_name: str, acts_array: np.ndarray) -> None:
        """
        Detects dead neurons in the activation layer by measuring the ratio of neurons that always produce a zero
        activation. If the ratio exceeds a predefined threshold, it is considered that the activation layer has dead
        neurons.
        Args:
            acts_name (str): The name of the activation layer to be validated. The name should include the name of the
            activation function used, such as "Relu", "Tanh", "Sigmoid", etc.
            acts_array (np.ndarray): The activations obtained from the specified activation layer.
        Returns:
            None
        """
        if self.config.Dead.disabled:
            return
        acts_array = transform_2d(acts_array, keep="last")
        major_values = np.percentile(
            np.abs(acts_array), q=self.config.Dead.act_maj_percentile, axis=0
        )
        dead_count = np.count_nonzero(major_values < self.config.Dead.act_min_thresh)
        dead_ratio = dead_count / major_values.shape[0]
        if dead_ratio > self.config.Dead.neurons_ratio_max_thresh:
            main_msg = self.main_msgs["act_dead"]
            self.error_msg.append(
                main_msg.format(dead_count, major_values.size, acts_name)
            )

    def check_acts_distribution(self, acts_name: str, acts_array: np.ndarray) -> None:
        """
        Checks the stability of the activation values over multiple iterations. To do this we watch the histograms of
        sampled activation layer while expecting to have normally-distributed values with unit standard deviation,
         e.g., a value within [0.5, 2]. the test would pass the actual std belongs to the range of [0.5, 2]; otherwise,
         we perform an f-test to compare std with either the low-bound 0.5 if std < 0.5 or 2.0 if std > 2.0.
        Args:
            acts_name (str): The name of the activation layer to be validated. The name should include the name of the
            activation function used, such as "Relu", "Tanh", "Sigmoid", etc.
            acts_array (np.ndarray): The activations obtained from the specified activation layer.
        Returns:
            None
        """
        if self.config.Distribution.disabled:
            return
        acts_array = transform_2d(acts_array, keep="last")
        act_std = np.std(acts_array)
        if (
            act_std < self.config.Distribution.std_acts_min_thresh
            or act_std > self.config.Distribution.std_acts_max_thresh
        ):
            if act_std < self.config.Distribution.std_acts_min_thresh:
                f_test_result = pure_f_test(
                    acts_array,
                    self.config.Distribution.std_acts_min_thresh,
                    self.config.Distribution.f_test_alpha,
                )
            else:
                f_test_result = pure_f_test(
                    acts_array,
                    self.config.Distribution.std_acts_max_thresh,
                    self.config.Distribution.f_test_alpha,
                )
            if not (f_test_result[1]):
                main_msg = self.main_msgs["act_unstable"]
                self.error_msg.append(
                    main_msg.format(
                        acts_name,
                        act_std,
                        self.config.Distribution.std_acts_min_thresh,
                        self.config.Distribution.std_acts_max_thresh,
                    )
                )

    def update_buffer(self, acts_name: str, acts_array: torch.Tensor) -> np.ndarray:
        """
        Updates the buffer for a given activation layer with new activations' data.
        Args:
        acts_name (str): The name of the activation layer to be validated. The name should include the name of the
        activation function used, such as "Relu", "Tanh", "Sigmoid", etc.
        acts_array (Tensor): The activations obtained from the specified activation layer.
        Returns:
        (numpy.ndarray) : The updated buffer containing the activations' data of the specified activation layer.
        """
        n = acts_array.shape[0]
        self._acts_data[acts_name][0:-n] = self._acts_data[acts_name][
            -(self.config.buff_scale - 1) * n :
        ]
        self._acts_data[acts_name][-n:] = acts_array.cpu().detach().numpy()
        return self._acts_data[acts_name]

    def set_acts_data(self, observations_size: int, activations: dict) -> None:
        """
        Creates the buffer that will contain the activations' history.
        Args:
        observations_size (str): size of the observations.
        activations (dict): dict of activation values.
        Returns:
        numpy.ndarray: The updated buffer containing the activations' history.
        """
        acts_data = {}
        for i, (acts_name, acts_array) in enumerate(activations.items()):
            acts_name = re.sub(r"\([^()]*\)", "", str(acts_name)) + "_" + str(i)
            dims = [int(dim) for dim in acts_array.shape[1:]]
            buffer_size = self.config.buff_scale * observations_size
            acts_data[acts_name] = np.zeros(shape=(buffer_size, *dims))
        self._acts_data = acts_data
