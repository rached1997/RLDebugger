# DRLDebugger

## Introduction to the DRL Debugger Tool

This package provides a debugging tool for Deep Reinforcement Learning (DRL) frameworks,
designed to detect and address various issues that may arise during training. 
The tool allows you to monitor your training process in real-time, identifying any 
potential flaws and making it easier to improve the performance of your DRL models.

### Checkers Included in this Package

> Note : This first version of the debugger contains checks for 
various aspects of the neural network, as adapted from the paper
"Testing Feedforward Neural Networks Training Programs" [^1]. 
We would like to thank this paper's authors for providing the code 
for these checks (https://github.com/thedeepchecker/thedeepchecker). 
These checks have been adapted to function in the DRL context and 
been migrated from TensorFlow 1 to PyTorch.

[^1]: https://arxiv.org/pdf/2204.00694.pdf

[//]: # (For more information on the specific Checkers included in this )

[//]: # (package, please refer to the [Debugger Details]&#40;./Debugger.md&#41; )

[//]: # (section.)

## Integrating the Debugger tool in your DRL Pytorch Application
To integrate the tool you have to do the following 4 steps : 

### Step 1. Prepare the config file
***

Create a .yml config file and copy the following lines :
```yml
debugger:
  name: 'Debugger'
  kwargs:
    
    params:
      observations: "variable" # sample of observations.
      model: "variable"        # Model to be trained.
      targets: "variable"      # Ground truth to be used for the Loos function .
      predictions: "variable"  # The outputs of the model in the sample of observations.
      loss_fn: "constant"      # Loss function.
      opt: "constant"          # Optimizer function.
      actions: "variable"      # Predicted actions for the sample of observations.
      done: "variable"         # Boolean indicating if the episode is done or not
      
    check_type:
      - name: #Checker_Name_1
      - name: #Checker_Name_2
```

The debugger config should have the same structure and includes the following elements:

* **params (only change it when you will develop a new Checker)** : contains the elements 
required for the Checker. The names of the parameters should not be changed, as the Checkers
will track these variables using the same names provided in this code snippet. constant or 
variable indicates if the nature of the variable.
* **check_type** : Mention the name of the check you want to do, by replacing 
`#Checker_Name_1` and `#Checker_Name_2` (you can add as many Checkers as you want). The names
of the Checker should be one of the following: `PreTrainObservation`, `PreTrainWeight`,
`PreTrainBias`, `PreTrainLoss`, `PreTrainProperFitting`, `PreTrainGradient`, `OnTrainLoss`, 
`OnTrainBias`, `OnTrainWeight`, `OnTrainActivation`. Be careful when choosing the name, 
it should be one of the checks listed.

### 2. Installation and Importing
***

> Please note that this step is temporary, as the project is still in development.

##### To set up the debugger in your python environment, follow these steps:
1. Clone the repository
2. cd 'path to RLDebugger repo'
3. Run the command `pip install -e .`
4. Import the debugger in your code with the following line:
```python
from debugger import rl_debugger
```
4. Set up the configuration using the following line :
```python
rl_debugger.set_config(config_path="the path to your debugger config.yml file")
```

### 3. Configuring each Checker (Optional)
***

If you only need to deactivate a specific check for a particular Checker, this step is useful.
You can modify the configuration of each Checker by changing the parameters in the
get_config function.

For instance, when debugging the weights during training (i.e `OnTrainWeight` Checker),
three different types of checks can be performed, but you may want to modify the thresholds or
deactivate a specific check. To do so, all you have to do is
modify the parameters you find in the config dict in the  function get_config function,
which you will find in the class of the targeted Checker, as shown below :
```python
def get_config():
    config = {
        "start": 3,
        "Period": 3,
        "numeric_ins": {"disabled": False},
        "neg": {"disabled": False, "ratio_max_thresh": 0.95},
        "dead ": {"disabled": False, "val_min_thresh": 0.00001, "ratio_max_thresh": 0.95},
        "div": {"disabled": False, "window_size": 5, "mav_max_thresh": 100000000, "inc_rate_max_thresh": 2}
    }
    return config
```


### 4. Run the debugging
***

To run the debugging, use the following code:

```python
from debugger import rl_debugger
....
rl_debugger.run_debugging(observations= ...,
                          model= ...,
                          targets= ...,
                          predictions= ....
                          )
```
This function will run the Checkers shosen by the user in the config file. 
It can be called from any class or file in your project ( you can imagine it as a
global function).

For example, to run the Checker PreTrainLoss, the `run_debugging` function should receive
four parameters: targets, predictions, loss_fn, and model. However,The function 
`run_debugging` can be called  in different parts of the code and be provided with 
the parameters available at that point, but the Checker will wait until all the 
parameters are received, so it can start running.

It is important to note that while calling `run_debugging`, the key for the parameters 
(args) must match exactly as mentioned in the configuration file under `params`. If there 
are parameters that are not required (i.e., not used by any of the Checkers), 
they can be omitted.

### 5. how to interpret the results
When you run the training, the debugging process will generate warning messages 
indicating any errors that have occurred and the elements that caused them.
To help you better understand the results of the debugging process, it's important
to carefully review these messages and take the necessary actions to resolve any 
issues that they highlight.

## Creating a new Checker

### 1. Create the Checker Class
***
To create a new checker, you can follow the structure outlined in the code snippet below:

```python
from debugger import DebuggerInterface


def get_config():
    config = {"Period": ..., "other_data": ...}
    return config


class CustomChecker(DebuggerInterface):
    def __init__(self):
        super().__init__(check_type="CustomChecker", config=get_config())
        # you can add other attributes

    #  You can define other functions

    def run(self, observed_param):
        if self.check_period():
            # Do some instructions ....
            self.error_msg.append("your error message")

```

To create a new Checker, you need to include the following elements in your class:
1. `get_config` function: This function is mandatory and defines all the configurations 
necessary for running your custom Checker. In the `config` dictionary, it is important to
include the `period` element, which determines the periodicity of the debugging 
(if you want the Checker to run only before the training, set its value to 0).

2. Your Checker class: Your Checker should inherit from the `DebuggerInterface` class and
initialize itself by calling `super()` and providing the name of your Checker
of your Checker 

3. The `run` function: The function should include three important elements: 
the parameters you need (as mentioned in the `params` in the config.yml file), 
the periodicity check using the predefined `check_period()` function, 
and appending the messages you want to display to the `self.error_msg list.`

Note: If you need to know the number of times the Checker has run, you can check 
it using the `self.iter_num` variable.

### 2. Register the Checker
***
To run your new Checker, you need to register it by adding the following line
in your `main.py`:

```python
rl_debugger.register(checker_name="CustomChecker", checker_class=CustomChecker)
# the register method should be called before the set_config method
rl_debugger.set_config(...)
```
Finally, to run your Checker, all you have to do is add "CustomChecker"
to the `config.yml` file and run the training.
