# TorchBricksRL

## Prerequisites
### Enable web Bluetooth on chrome

 1. Go to "chrome://flags/" 
 2. enable "Experimental Web Platform features"
 3. restart chrome
 4. Use beta.pybricks.com to edit and upload the client scripts for each environment

### Environment Setup

1. **Create a Conda environment:**
   ```bash
   conda create --name torchbricksrl python=3.8
    ```
2. **Activate the environment:**
   ```bash
   conda activate torchbricksrl
    ```
3. **Install PyTorch:**
   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

    ```
4. **Install additional packages:**
   ```bash
   pip install -r requirements.txt
    ```

## Usage
### Client 
Update your client script on the [PyBricks Hub](https://beta.pybricks.com/) whenever you want to run a new environment

### Repo Structure
<details>
  <summary>Click me</summary>

```
project_root/
│
├── configs/                    # Centralized configuration directory
│   ├── config.yaml             # Base config
│   ├── env/                    # Environment and task specific configs
|   |   ├── runaway-v0.yaml
|   |   ├── spinning_v0.yaml
|   |   ├── walker-v0.yaml
|   |   ├── walker_sim-v0.yaml
|   |   ├── roboarm-v0.yaml
|   |   ├── roboarm_sim-v0.yaml
|   |   └── roboarm_mixed-v0.yaml  
│   └── agent/                   # Agent specific configs
|       ├── sac.yaml
|       ├── td3.yaml
|       └── droq.yaml
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
│   └── ...
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
</details>

### Config
Before running experiments, please review and modify the configuration settings according to your needs. Each environment and agent setup has its own specific configuration file under the configs/ directory. For more information checkout the [config README](conf/README.md).

## Run Experiments
### Train an Agent

   ```bash
   python experiments/walker/train.py
   ```

### Evaluate an Agent
   ```bash
   python experiments/walker/eval.py
   ```

## Results
### 2Wheeler Results:

<img src="https://drive.google.com/uc?export=view&id=1U2s_zKFJyHdb1EnetHSpmB7DHs9Tz-vG" width="500" height="300" alt="Training Curve">




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