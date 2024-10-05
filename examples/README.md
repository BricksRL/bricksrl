# Examples 

## TorchRL SOTA Example

In the [torchrl_sac](./torchrl_sac/) folder you will find a training script to train LEGO robots with Bricksrl similar to the state-of-the-art implementations in [TorchRL](https://github.com/pytorch/rl/tree/main/sota-implementations). This allows you to basically plug-and-play with any TorchRL sota-implementation or do custom adaptations.

[Example results](https://wandb.ai/sebastian-dittert/bricksrl_torchrl_sac_example?nw=nwusersebastiandittert)

### TorchRL sota-example diff

Only change made to the TorchRL sota-implementations is the make_env function:

```
# Environment import from BricksRL
from bricksrl.environments.walker_v0.WalkerEnvSim import WalkerEnvSim_v0

# ====================================================================
# Make BricksRL Environment
# -----------------


def env_maker(cfg, device="cpu", from_pixels=False):
    # We use the WalkerEnvSim_v0 environment from BricksRL as an example
    # as it is easy to test as it does not require a robot at hand or to connect to the hub.
    # Users can replace this with any other environment from BricksRL or custom environments.
    env = WalkerEnvSim_v0(max_episode_steps=cfg.env.max_episode_steps)
    observation_keys = [key for key in env.observation_spec.keys()]

    transforms = []
    if cfg.env.frame_stack > 1:
        transforms.append(
            CatFrames(
                N=cfg.env.frame_stack,
                in_keys=observation_keys,
                out_key=observation_keys,
            )
        )
    normalize_keys = [key for key in observation_keys if key != "pixels"]
    obs_ranges = np.array(list(env.observation_ranges.values()))
    obs_mean = obs_ranges.mean(axis=-1)
    obs_std = obs_ranges.std(axis=-1)
    transforms.append(
        ObservationNorm(
            in_keys=normalize_keys, loc=obs_mean, scale=obs_std, standard_normal=True
        )
    )
    transforms.append(DeviceCastTransform(device))
    return TransformedEnv(env, Compose(*transforms))

```




## Custom Environment 
A template to create your own custom environments can be found [here](./custom_env.py). With an environment created like this you can update the [TorchRL example](./torchrl_sac) to train your own TorchRL agent on your custom environment.

## High Level Examples
In the [example notebook](./example_notebook.ipynb) we provide high-level training examples to train a **SAC agent** in the **RoboArmSim-v0** environment and a **TD3 agent** in the **WalkerSim-v0** enviornment.
The examples are based on the experiments for our paper. Stand alone examples similar to the [TorchRL sota-implementations](https://github.com/pytorch/rl/tree/main/sota-implementations) can be found [here](./torchrl_sac).