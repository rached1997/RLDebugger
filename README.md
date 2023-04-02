# DRLDebugger


[<img src="https://img.shields.io/badge/license-Apache 2.0-blue">](https://github.com/rached1997/RLDebugger)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This package provides a debugging tool for Deep Reinforcement Learning (DRL) frameworks,
designed to detect and address DNN and RL issues that may arise during training. 
The tool allows you to monitor your training process in real-time, identifying any 
potential flaws and making it easier to improve the performance of your DRL models.

The implementation is clean and simple, with research-friendly features. The highlight features of DRLDebugger are:

* 📜 Straightforward integration
   * DRLDebugger can be integrated into your project with a few lines of code.
* 🗳️ DNN + RL checks
* 🛃 Custom checks 
* 🖥️ Real-time warnings
* 📈 Monitoring using [Weights and Biases](https://wandb.ai/site) 


## Get started

Prerequisites:
* Python >=3.7.7,<3.10 (not yet tested on 3.10)

Step 1. Install the debugger in your python environment:
```bash
git clone https://github.com/rached1997/RLDebugger.git && cd RLDebugger
pip install -e .
```

Step 2. Create a .yml config file and copy the following lines:
```yml
debugger:
  name: 'Debugger'
  kwargs:
    observed_params:
      constant: []
      variable: []
    check_type:
      - name: #Checker_Name_1
      - name: #Checker_Name_2
```

Step 3. Set up the config and run the debugger:

```python
from debugger import rl_debugger

rl_debugger.set_config(config_path="the path to your debugger config.yml file")

...

rl_debugger.debug(model=...,
                  max_total_steps=...,
                  targets=...,
                  action_probs=....
                  )
```

For detailed steps on how to integrate the debugger, please refer to 
[Integrating the Debugger](#integrating-the-debugger-tool-in-your-drl-pytorch-application)

## Checkers Included in this Package

> The checkers included in this pachage are divided into two categories:
DNN-specific checkers and RL-specific checkers. The DNN-specific checkers
were adapted from the paper "Testing Feedforward Neural Networks Training 
Programs" [^1]. We would like to thank this paper's authors for providing the 
code for these checks (https://github.com/thedeepchecker/thedeepchecker). 
These checks have been adapted to function in the DRL context and been 
migrated from TensorFlow 1 to PyTorch. The following Table list all the checkers
with a link to their location in the package (you can find there more details on
each checker).

| Category   | Check                | Description                                                                                                                        |
|------------|----------------------|------------------------------------------------------------------------------------------------------------------------------------|
| DNN Checks | Activation           | [Link to Description](https://github.com/rached1997/RLDebugger/blob/dev/debugger/checkers/nn_checkers/activation_check.py)         |
|            | Loss                 | [Link to Description](https://github.com/rached1997/RLDebugger/blob/dev/debugger/checkers/nn_checkers/loss_check.py)               |
|            | Weight               | [Link to Description](https://github.com/rached1997/RLDebugger/blob/dev/debugger/checkers/nn_checkers/weight_check.py)             |
|            | Bias                 | [Link to Description](https://github.com/rached1997/RLDebugger/blob/dev/debugger/checkers/nn_checkers/bias_check.py)               |
|            | Gradient             | [Link to Description](https://github.com/rached1997/RLDebugger/blob/dev/debugger/checkers/nn_checkers/gradient_check.py)           |
|            | ProperFitting        | [Link to Description](https://github.com/rached1997/RLDebugger/blob/dev/debugger/checkers/nn_checkers/proper_fitting_check.py)     |
| DRL Checks | Action               | [Link to Description](https://github.com/rached1997/RLDebugger/blob/dev/debugger/checkers/rl_checkers/action_check.py)             |
|            | Agent                | [Link to Description](https://github.com/rached1997/RLDebugger/blob/dev/debugger/checkers/rl_checkers/agent_check.py)              |
|            | Environment          | [Link to Description](https://github.com/rached1997/RLDebugger/blob/dev/debugger/checkers/rl_checkers/environment_check.py)        |
|            | ExplorationParameter | [Link to Description](https://github.com/rached1997/RLDebugger/blob/dev/debugger/checkers/rl_checkers/exploration_param_check.py)  |
|            | Reward               | [Link to Description](https://github.com/rached1997/RLDebugger/blob/dev/debugger/checkers/rl_checkers/reward_check.py)             |
|            | Step                 | [Link to Description](https://github.com/rached1997/RLDebugger/blob/dev/debugger/checkers/rl_checkers/steps_check.py)              |
|            | State                | [Link to Description](https://github.com/rached1997/RLDebugger/blob/dev/debugger/checkers/rl_checkers/states_check.py)             |
|            | UncertaintyAction    | [Link to Description](https://github.com/rached1997/RLDebugger/blob/dev/debugger/checkers/rl_checkers/uncertainty_action_check.py) |
|            | ValueFunction        | [Link to Description](https://github.com/rached1997/RLDebugger/blob/dev/debugger/checkers/rl_checkers/value_function_check.py)     |

[^1]: https://arxiv.org/pdf/2204.00694.pdf

[//]: # (For more information on the specific Checkers included in this )

[//]: # (package, please refer to the [Debugger Details]&#40;./Debugger.md&#41; )

[//]: # (section.)

## Integrating the Debugger tool in your DRL Pytorch Application
To integrate the tool you have to do the following 4 steps : 

### 1. Prepare the config file
***

Create a .yml config file and copy the following lines :
```yml
debugger:
  name: 'Debugger'
  kwargs:
    observed_params:
      constant: []
      variable: []
    check_type:
      - name: #Checker_Name_1
        period: #Checker_period_value
        skip_run_threshold: #Checker_skip_run_value
      - name: #Checker_Name_2
```

The debugger config should have the same structure and includes the following elements:

* **observed_params (only add the two lists when you will develop a new Checker)** : contains the elements 
observed by the debugger. We have already a list of the default observed params (reward, actions, ...) so please 
only add non default params. The list of default params can be found [here](https://github.com/rached1997/RLDebugger/blob/dev/debugger/utils/config/default_debugger.yml).
Constant or variable indicates the nature of the observed parm.
* **check_type** : Mention the name of the check you want to perform, by replacing 
`#Checker_Name_1` and `#Checker_Name_2` (you can add as many Checkers as you want). The names
of the Checker can be found in the above table (Check Column).
  * **period** : You can specify the period over which the checker will be called each time. If you don't specify a 
  specific period the default values found in the [config data classes](https://github.com/rached1997/RLDebugger/tree/dev/debugger/config_data_classes) 
  will be automatically used.
  * **skip_run_threshold** : You can specify the number of skipped steps over which the checker will be called each 
  time. If you don't specify a specific value the default values found in the [config data classes](https://github.com/rached1997/RLDebugger/tree/dev/debugger/config_data_classes) 
  will be automatically used.

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
config data class or by creating your own instance.

For instance, when debugging the weights during training (i.e `Weight` Checker),
three different types of checks can be performed, but you may want to modify the thresholds or
deactivate a specific check. To do so, all you have to do is
modify the parameters you find in the `WeightConfig`,
which you will find [here](https://github.com/rached1997/RLDebugger/tree/dev/debugger/config_data_classes), as shown below :
```python
@dataclass
class WeightConfig:
    start: int = 100
    period: int = 10
    skip_run_threshold: int = 2
    numeric_ins: NumericIns = NumericIns()
    neg: Neg = Neg()
    dead: Dead = Dead()
    div: Div = Div()
    initial_weight: InitialWeight = InitialWeight()
```


### 4. Run the debugging
***

To run the debugging, use the following code:

```python
from debugger import rl_debugger

....
rl_debugger.debug(model=...,
                  max_total_steps=...,
                  targets=...,
                  action_probs=....
                  )
```
This function will run the Checkers chosen by the user in the config file. 
It can be called from any class or file in your project ( you can imagine it as a
global function).

For example, to run the Checker `Action`, the `debug` function should receive
three parameters: actions_probs, max_total_steps, and max_reward. Moreover,The function 
`debug` can be called  in different parts of the code and be provided with 
the parameters available at that point, but the Checker will wait until all the 
parameters are received, so it can start running.

It is important to note that the environment is the only parameter required for the debugger to properly operate. 
Also, the environment needs to be passed at the beginning of your code and in the first call of ".debug()".

```python
from debugger import rl_debugger

....
env = gym.make("CartPole-v1")
rl_debugger.debug(environment=env)
```

It is also important to note that while calling `debug`, the key for the parameters 
(args) must match exactly as mentioned in the configuration file under `observed_params`. If there 
are parameters that are not required (i.e., not used by any of the Checkers), 
they can be omitted.

### 5. how to interpret the results
When you run the training, the debugging process will generate warning messages 
indicating any errors that have occurred and the elements that caused them.
To help you better understand the results of the debugging process, it's important
to carefully review these messages and take the necessary actions to resolve any 
issues that they highlight.

### 6. data visualization
To help you better understand the results of the debugging process, we added visualization
options for some checkers (`Action` and `Reward` checkers) using `Wandb`. You can follow the 
same logic in your custom checkers if you want to add visualization to them. Please explore 
[here](https://github.com/rached1997/RLDebugger/blob/dev/debugger/utils/wandb_logger.py) for more details.

## Creating a new Checker

### 1. Create the Checker Class
***
To create a new checker, you can follow the structure outlined in the code snippet below:

```python
from debugger import DebuggerInterface
from dataclasses import dataclass

@dataclass
class CustomCheckerConfig:
    period: int = ...
    other_data: float = ...
    

class CustomChecker(DebuggerInterface):
    def __init__(self):
        super().__init__(check_type="CustomChecker", config=CustomCheckerConfig)
        # you can add other attributes

    #  You can define other functions

    def run(self, observed_param):
        if self.check_period():
            # Do some instructions ....
            self.error_msg.append("your error message")

```

To create a new Checker, you need to include the following elements in your class:
1. `CustomCheckerConfig` data class: This class is mandatory and defines all the configurations 
necessary for running your custom Checker. It is necessary to include the `period` element, 
which determines the periodicity of the debugging (if you want the Checker to run 
only before the training, set its value to 0).

2. Your Checker class: Your Checker should inherit from the `DebuggerInterface` class and
initialize itself by calling `super()` and providing the name of your Checker
of your Checker 

3. The `run` function: The function should include three important elements: 
the parameters you need (as mentioned in the `observed_params` in the config.yml file), 
the periodicity check using the predefined `check_period()` function, 
and appending the messages you want to display to the `self.error_msg list.`

Notes: 
* If you need to know the number of times the Checker has run, you can check 
it using the `self.iter_num` variable.
* If you need to know the number of steps the RL algorithm has performed, you can check 
it using the `self.step_num` variable.

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

## Important Notes

1. The environment is the only parameter required for the debugger to properly operate. Also, the environment needs 
to be passed at the beginning of your code and in the first call of ".debug()".

```python
from debugger import rl_debugger

....
env = gym.make("CartPole-v1")
rl_debugger.debug(environment=env)
```

2. Every observed parameters (e.g., model, target_model, action_probs, etc) need to be send once to the debugger through
the ".debug()".
The following code snippet is a wrong behavior.

```python
from debugger import rl_debugger

....
state, reward, done, _ = env.step(action)
qvals = qnet(state)
rl_debugger.debug(model=qnet)
...
batch = replay_buffer.sample(batch_size=32)
qvals = qnet(batch["state"])
rl_debugger.debug(model=qnet)
```

The above code would result in a wrong behavior as the model is sent twice to the debugger. You should avoid send 
the same observed parameters from two different code locations.

3. If you have a test run during the learning process, you have to turn off/on the debugger. Otherwise, 
some unexpected behavior may arise. 

```python
from debugger import rl_debugger

....
def run_testing():
        rl_debugger.turn_off()
        results = super().run_testing()
        rl_debugger.turn_on()
        return results
```
4. It is recommended to add all the constant observed params (e.g., max_reward, loss_fn, max_total_steps, ...) at the 
beginning of your code and in the first call of ".debug()". These params needs to be observed once and many checkers 
rely on them. Thus, providing them once and in at the beginning of your code would be more efficient.

```python
from debugger import rl_debugger

....
env = gym.make("CartPole-v1")
rl_debugger.debug(
        environment=env,
        max_reward=reward_threshold,
        max_steps_per_episode=max_steps_per_episode,
        max_total_steps=max_train_steps,
    )
```

## Support and get involved

Feel free to ask questions. Posting in [Github Issues](https://github.com/rached1997/RLDebugger/issues) and PRs are also 
welcome.

