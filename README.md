# RLDebugger

## Introducing the tool
### What the tool can do
### What checks the tool does


## Integrating the Debugger tool in your DRL framework
To integrate the tool you have to do the following 4 steps : 

### Step 1. Prepare the config file

***

Create a .yml config file and copy the following lines :
```yml
debugger:
  name: 'Debugger'
  kwargs:
    params:
      observations: 0
      model: 0
      labels: 0
      predictions: 0
      loss_fn: -1
      opt: -1
      actions: 0
      done: 0
    check_type:
      - name: #Debugger_Name_1
      - name: #Debugger_Name_2
```
The debugger config should have the same structure and includes the following elements:

* **params (only change it when you will develop a new debugger)** : contains the elements 
required for the debugger. The names of the parameters should not be changed, as the debuggers
will track these variables using the same names provided in this code snippet. constant or 
variable indicates if the nature of the variable.
* **check_type** : Mention the name of the check you want to do, by replacing 
`#Debugger_Name_1` and `#Debugger_Name_2` (you can add as many debuggers as you want). The names
of the debugger should be one of the following checks: `PreTrainObservation`, `PreTrainWeight`,
`PreTrainBias`, `PreTrainLoss`, `PreTrainProperFitting`, `PreTrainGradient`, `OnTrainLoss`, 
`OnTrainBias`, `OnTrainWeight`, `OnTrainActivation`. Be careful when choosing the name, 
it should be one of the checks listed.

### 2. Installation and Importing
***

> Please note that this step is temporary, as the project is still in development.

##### To set up the debugger in your python environment, follow these steps:
1. Clone the repository
2. Run the command pip install -e . in the terminal
3. Import the debugger in your code with the following line:
```python
from debugger import rl_debugger
```
4. Set up the configuration using the following line :
```python
rl_debugger.set_config(config_path="the path to your debugger config.yml file")
```

### 3. Configuring each Debugger (Optional)
***

If you only need to deactivate a specific check for a particular debugger, this step is useful.
You can modify the configuration of each debugger by changing the parameters in the
get_config function.

For instance, when debugging the weights during training (i.e `OnTrainWeight` debugger),
three different types of checks can be performed, but you may want to modify the thresholds or
deactivate a specific check. To do so, all you have to do is
modify the parameters you find in the config dict in the  function get_config function,
which you will find in the class of the targeted debugger, as shown below :
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
                          labels= ...,
                          predictions= ....
                          )
```
This function will run the debuggers shosen by the user in the config file. 
It can be called from any class or file in your project ( you can imagine it as a
global function).

For example, to run the debugger PreTrainLoss, the `run_debugging` function should receive
four parameters: labels, predictions, loss_fn, and model. However,The function 
`run_debugging` can be called  in different parts of the code and be provided with 
the parameters available at that point, but the debugger will wait until all the 
parameters are received, so it can start running.

It is important to note that while calling `run_debugging`, the key for the parameters 
(args) must match exactly as mentioned in the configuration file under `params`. If there 
are parameters that are not required (i.e., not used by any of the debugger), 
they can be omitted.

### 5. how to interpret the results
When you run the training, the debugging process will generate warning messages 
indicating any errors that have occurred and the elements that caused them.
To help you better understand the results of the debugging process, it's important
to carefully review these messages and take the necessary actions to resolve any 
issues that they highlight.

## Creating a new Debugger

### 1. Create the Debugger Class

To create a new debugger, you can follow the structure outlined in the code snippet below:

```python
def get_config():
    config = {"Period": ..., "other_data": ...}
    return config


class YourDebuggerName(DebuggerInterface):
    def __init__(self):
        super().__init__(check_type="YourDebuggerName", config=get_config())
        # you can add other attributes    
    
    #  You can define other functions
    
    def run(self, put_the_parameters_you_need):
        if not self.check_period():
            # Do some instructions ....
        
        # You can do your debugging logic here ....
        
        self.error_msg.append("your error message")
```

To create a new debugger, you need to include the following elements in your class:
1. `get_config` function: This function is mandatory and defines all the configurations 
necessary for running your custom debugger. In the `config` dictionary, it is important to
include the `period` element, which determines the periodicity of the debugging 
(if you want the debugger to run only before the training, set its value to 0).

2. Your debugger class: Your debugger should inherit from the `DebuggerInterface` class and
initialize itself by calling `super()` and providing the name of your debugger
of your debugger 

3. The `run` function: The function should include three important elements: 
the parameters you need (as mentioned in the `params` in the config.yml file), 
the periodicity check using the predefined `check_period()` function, 
and appending the messages you want to display to the `self.error_msg list.`

Note: If you need to know the number of times the debugger has run, you can check 
it using the `self.iter_num` variable.

### Register the Debugger
To run your new debugger, you need to register it by adding the following line
in the `__init__.py` file under the debugger:

```python
registry.register("YourDebuggerName", YourDebuggerName, YourDebuggerName)
```
Finally, to run your debugger, all you have to do is add "YourDebuggerName"
to the `config.yml` file and run the training.