# Configuration Details for TorchBricksRL Experiments

## Overview
This directory contains all the necessary configuration files to tailor your experiments using PyBricksRL. Configurations are managed using [Hydra](https://hydra.cc/), a powerful tool for configuring complex applications that allows for easy modification of parameters directly from the command line.

## Configuration Files
- **config.yaml**: The base configuration for all experiments including what agent and environment to run. 
- **env/**: Contains environment-specific configurations.
  - **runaway-v0.yaml**: Settings for the *RunAway-v0* environment for the 2wheeler robot.
  - **spinning-v0.yaml**: Settings for the *Spinning-v0* environment for the 2wheeler robot.
  - **walker-v0.yaml**: Settings for the *Walker-v0* environment for the walker robot.
  - **walker_sim-v0.yaml**: Settings for the *WalkerSim-v0* environment for the walker robot.
  - **roboarm-v0.yaml**: Settings for the *RoboArm-v0* environment for the roboarm robot.
  - **roboarm_sim-v0.yaml**: Settings for the *RoboArmSim-v0* environment for the roboarm robot.
  - **roboarm_mixed-v0.yaml**: Settings for the *RoboArmMixed-v0* environment for the roboarm robot.
- **agent/**: Contains agent-specific configurations.
  - **sac.yaml**: Configuration for the SAC agent.
  - **td3.yaml**: Configuration for the TD3 agent.
  - **droq.yaml**: Configuration for the DroQ agent.

## Using Hydra for Configuration Overrides
Hydra allows you to override any configuration parameter directly from the terminal when you run your experiments. This makes it easy to test different configurations without altering your configuration files.

### Example Usage
To run an experiment with the walker environment using the SAC agent and specify the number of episodes directly from the command line, you can use the following command:

```bash
python experiments/walker/train.py episodes=200 agent=sac
```
This command temporarily overrides the episodes and agent parameters for this specific run without needing to change the configuration files.

You can further override agent or environment specific parameter like:

```bash
python experiments/walker/train.py agent=sac agent.batch_size=32
```

or 

```bash
python experiments/walker/train.py env.max_episode_steps=200 env.frame_stack=4
```