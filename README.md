# PyBricksRL

## Enable web Bluetooth on chrome

 1. Go to "chrome://flags/" 
 2. enable "Experimental Web Platform features"
 3. restart chrome
 4. Use beta.pybricks.com to edit and upload the client scripts for each environment

## Setup Environment and Packages

Create your conda environment with: `conda create --name lego python=3.8`

Activate the conda environment with: `conda activate lego`

Install PyTorch: `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

Install packages needed with: `pip install -r requirements.txt`


## Client 
Update your client script on the [PyBricks Hub](https://beta.pybricks.com/) whenever you want to run a new environment


## Helpers:
We created a [PyBricks GPT](https://chat.openai.com/g/g-YZYA1j4yF-pybrics-lego-doc) which is an expert on the PyBricks documentation and can help you with coding and debugging client code for the PyBricks Hub.


# Additional Info
Code examples from PyBricks. Might be interesting as expert policies for offline RL. [GitHub Pybricks examples](https://github.com/pybricks/pybricks-projects/blob/master/sets/mindstorms-robot-inventor/main-models/gelo/gelo_basic.py)




## Repo structure 

```
project_root/
│
├── configs/                    # Centralized configuration directory
│   ├── config.yaml             # Base config
│   ├── env/                    # Environment and task specific configs
|   |   ├── runaway_v0.yaml
|   |   ├── walker_v0.yaml
|   |   └── roboarm_v0.yaml  
│   └── agent/                   # Agent specific configs
|       ├── sac.yaml
|       ├── td3.yaml
|       └── drq.yaml
│   
├── experiments/                # Experiments directory
│   ├── 2wheeler/               # 2wheeler robot specific experiments
|   |   ├── train.py
|   |   └── eval.py  
│   ├── walker/                 # Walker robot specific experiments
|   |   ├── train.py
|   |   └── eval.py  
│   └── roboarm/                # Roboarm specific experiments
|       ├── train.py
|       └── eval.py  
│
├── environments/               # Environments directory
│   ├── __init__.py
│   ├── base/                   # Base environment class
|   |   ├── base_env.py
|   |   └── PybricksHubClass.py # For Async-Communication with the robot
│   ├── runaway_v0.py           # Environment for the 2wheeler robot
|   |   ├── client.py
|   |   └── Env.py  
│   ├── walker_v0.py            # Environment for the walker
|   |   ├── client.py
|   |   └── Env.py
│   └── roboarm_v0.py           # Environment for the roboarm
|       ├── client.py
|       └── Env.py
│
├── src/                     # Source code for common utilities, robot models, etc.
│   ├── __init__.py
│   ├── utils/
│   ├── agents/
|   |   ├── sac.py
|   |   └── td3.py
│   └── networks/
|       └── ...
│
└── tests/                   # Unit tests and integration tests
    ├── ...

```