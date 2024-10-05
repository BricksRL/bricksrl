# BricksRL

![CI](https://github.com/BricksRL/bricksrl/actions/workflows/ci.yaml/badge.svg?branch=main)
![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9-blue)
[![arXiv](https://img.shields.io/badge/arXiv-2406.17490-b31b1b.svg)](https://arxiv.org/abs/2406.17490)
[![Website](https://img.shields.io/badge/Website-Visit%20Now-blue)](https://bricksrl.github.io/ProjectPage/)
[![Discord](https://img.shields.io/badge/Join_our_Discord-7289da?logo=discord&logoColor=ffffff&labelColor=7289da)](https://discord.gg/qdTsFaVfZm)


BricksRL allows the training of custom LEGO robots using deep reinforcement learning. By integrating [PyBricks](https://pybricks.com/) and [TorchRL](https://pytorch.org/rl/stable/index.html), it facilitates efficient real-world training via Bluetooth communication between LEGO hubs and a local computing device. Check out our [paper](https://arxiv.org/abs/2406.17490)!

For additional information and building instructions for the robots, view the project page [BricksRL](https://bricksrl.github.io/ProjectPage/).




## Prerequisites
<details>
  <summary>Click me</summary>

### Enable web Bluetooth on chrome

 1. Go to "chrome://flags/" 
 2. enable "Experimental Web Platform features"
 3. restart chrome
 4. Use beta.pybricks.com to edit and upload the client scripts for each environment

### Environment Setup

1. **Create a Conda environment:**
   ```bash
   conda create --name bricksrl python=3.8
    ```
2. **Activate the environment:**
   ```bash
   conda activate bricksrl
    ```
3. **Install PyTorch:**
   ```bash
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

    ```
4. **Install bricksrl and additional packages:**
   For regular users, install the package and all required dependencies by running:
   ```bash
   pip install -e .
    ```

   This will install the bricksrl package along with the dependencies listed in setup.py.

5. **(Optional) Install development tools:**

   If you are a developer and need to install development tools (e.g., pytest, ufmt, pre-commit), use the following command to install them as extras:

   ```bash
   pip install -e .[dev]
   ```
   This will install the development dependencies defined in the setup.py file along with the package.


</details>

## Usage
### Client 
Update your client script on the [PyBricks Hub](https://beta.pybricks.com/) whenever you want to run a new environment with your robot.


### Config
Before running experiments, please review and modify the configuration settings according to your needs. Each environment and agent setup has its own specific configuration file under the configs/ directory. For more information checkout the [config README](conf/README.md).


### Robots

Robots utilized for our experiments. Building instructions can be found [here](https://bricksrl.github.io/ProjectPage/).

| ![2wheeler](https://drive.google.com/uc?export=view&id=1IxqQ1VZchPZMNXyZnTULuNy53-LMYT6W) | ![Walker](https://drive.google.com/uc?export=view&id=1ImR0f1UNjC4sUHXWWg_D06eukrh-doW9) | ![RoboArm](https://drive.google.com/uc?export=view&id=1IYCJrl5rZBvOb6xKwbSUZqYrVwKjCpJH) | 
|:--:|:--:|:--:|  
| **2Wheeler** |  **Walker** | **RoboArm** |


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
<details>
  <summary>Click me</summary>

Evaluation videos of the trained agents can be found [here](https://bricksrl.github.io/ProjectPage/).

### 2Wheeler Results:

<img src="https://drive.google.com/uc?export=view&id=1U2s_zKFJyHdb1EnetHSpmB7DHs9Tz-vG" width="500" height="300" alt="2Wheeler Results">

### Walker Results:

<img src="https://drive.google.com/uc?export=view&id=19ygYa7gBj8WBzn4ZwXc6007hyBDqTwJ6" width="500" height="300" alt="Walker Results">

### RoboArm Results:

<img src="https://drive.google.com/uc?export=view&id=1Qow1VICXMv25gMSP4Kt-fW37Kxl6rOWi" width="500" height="300" alt="RoboArm Results">

<img src="https://drive.google.com/uc?export=view&id=1d58vv2JNNP1U1x_oh8uz93wH8WKj3F4k" width="500" height="300" alt="RoboArm Mixed Results">

</details>


### Offline RL 
<details>
  <summary>Click me</summary>

With the use of precollected [datasets](https://huggingface.co/datasets/compsciencelab/BricksRL-Datasets) we can pretrain agents with offline RL to perform a task without the need of real world interaction. Such pretrained policies can be evaluated directly or used for later training to fine tuning the pretrained policy on the real robot. 

#### Datasets
The datasets can be downloaded from huggingface and contain expert and random transitions for the 2Wheeler (RunAway-v0 and Spinning-v0), Walker (Walker-v0) and RoboArm (RoboArm-v0) robots.

   ```bash
      git lfs install
      git clone git@hf.co:datasets/compsciencelab/BricksRL-Datasets
   ```

The datasets consist of TensorDicts containing expert and random transitions, which can be directly loaded into the replay buffer. When initiating (pre-)training, simply provide the path to the desired TensorDict when prompted to load the replay buffer.


#### Pretrain an Agent

The execution of an experiment for offline training is similar to the online training except that you run the **pretrain.py** script: 

   ```bash
   python experiments/walker/pretrain.py
   ```

Trained policies can then be evaluated as before with:

   ```bash
   python experiments/walker/eval.py
   ```

Or run training for fine-tuning the policy on the real robot:

   ```bash
   python experiments/walker/train.py
   ```


</details>

## High-Level Examples
In the [example notebook](examples/example_notebook.ipynb) we provide high-level training examples to train a **SAC agent** in the **RoboArmSim-v0** environment and a **TD3 agent** in the **WalkerSim-v0** enviornment. 
The examples are based on the experiments for our paper. Stand alone examples similar to the [TorchRL sota-implementations](https://github.com/pytorch/rl/tree/main/sota-implementations) can be found [here](examples/torchrl_sac).

## Citation
If you use BricksRL in your work, please refer to this BibTeX entry to cite it:

```
@article{dittert2024bricksrl,
  title={BricksRL: A Platform for Democratizing Robotics and Reinforcement Learning Research and Education with LEGO},
  author={Sebastian Dittert and Vincent Moens and Gianni De Fabritiis},
  journal={arXiv preprint arXiv:2406.17490},
  year={2024}
}
```