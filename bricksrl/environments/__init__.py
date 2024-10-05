import numpy as np
from torchrl.envs import (
    CatFrames,
    Compose,
    ObservationNorm,
    ToTensorImage,
    TransformedEnv,
)

from bricksrl.environments.roboarm_mixed_v0.RoboArmMixedEnv import RoboArmMixedEnv_v0
from bricksrl.environments.roboarm_v0.RoboArmEnv import RoboArmEnv_v0
from bricksrl.environments.roboarm_v0.RoboArmSim import RoboArmSimEnv_v0
from bricksrl.environments.runaway_v0.RunAwayEnv import RunAwayEnv_v0
from bricksrl.environments.spinning_v0.SpinningEnv import SpinningEnv_v0
from bricksrl.environments.walker_v0.WalkerEnv import WalkerEnv_v0
from bricksrl.environments.walker_v0.WalkerEnvSim import WalkerEnvSim_v0

VIDEO_LOGGING_ENVS = ["roboarm_mixed-v0", "walker_mixed-v0"]
ALL_2WHEELER_ENVS = ["spinning-v0", "runaway-v0"]
ALL_WALKER_ENVS = [
    "walker-v0",
    "walker_sim-v0",
]
ALL_ROBOARM_ENVS = [
    "roboarm-v0",
    "roboarm_mixed-v0",
    "roboarm_sim-v0",
]
ALL_ENVS = ALL_2WHEELER_ENVS + ALL_WALKER_ENVS + ALL_ROBOARM_ENVS


# TODO: maybe outsorce this to experiments/helper and not in bricksrl
def make_env(config, pretrain=False):
    """
    Creates a new environment based on the provided configuration.

    Args:
        config: A configuration object containing the environment name and maximum episode steps.
        pretrain: A boolean indicating whether the environment is for pretraining.

    Returns:
        A tuple containing the new environment, its action spec, and its state spec.
    """
    env = make(name=config.env.name, env_conf=config.env, pretrain=pretrain)
    observation_keys = [key for key in env.observation_spec.keys()]

    transforms = []
    if config.env.frame_stack > 1:
        transforms.append(
            CatFrames(
                N=config.env.frame_stack,
                in_keys=observation_keys,
                out_key=observation_keys,
            )
        )
    if config.env.action_filter < 1:
        raise NotImplementedError("ActionFilterWrapper not implemented yet")
        # TODO: add this to torchrl
        # env = ActionFilterWrapper(
        #     env, current_action_influence=config.env.action_filter
        # )
    normalize_keys = [key for key in observation_keys if key != "pixels"]
    obs_ranges = np.array(list(env.observation_ranges.values()))
    obs_mean = obs_ranges.mean(axis=-1)  # mean of min and max
    obs_std = obs_ranges.std(axis=-1)  # std of min and max
    transforms.append(
        ObservationNorm(
            in_keys=normalize_keys, loc=obs_mean, scale=obs_std, standard_normal=True
        )
    )
    if "pixels" in observation_keys:
        transforms.append(ToTensorImage(in_keys=["pixels"], from_int=True))

    env = TransformedEnv(env, Compose(*transforms))

    action_spec = env.action_spec
    state_spec = env.observation_spec

    return env, action_spec, state_spec


def make(name="RunAway", env_conf=None, pretrain=False):
    if name == "runaway-v0":
        return RunAwayEnv_v0(
            max_episode_steps=env_conf.max_episode_steps,
            min_distance=env_conf.min_distance,
            verbose=env_conf.verbose,
            pretrain=pretrain,
        )
    elif name == "spinning-v0":
        return SpinningEnv_v0(
            max_episode_steps=env_conf.max_episode_steps,
            sleep_time=env_conf.sleep_time,
            verbose=env_conf.verbose,
            pretrain=pretrain,
        )
    elif name == "walker-v0":
        return WalkerEnv_v0(
            max_episode_steps=env_conf.max_episode_steps,
            verbose=env_conf.verbose,
            sleep_time=env_conf.sleep_time,
            pretrain=pretrain,
        )
    elif name == "walker_sim-v0":
        return WalkerEnvSim_v0(
            max_episode_steps=env_conf.max_episode_steps,
            noise=env_conf.noise,
            low_action_angle=env_conf.low_action_angle,
            high_action_angle=env_conf.high_action_angle,
            verbose=env_conf.verbose,
        )
    elif name == "roboarm-v0":
        return RoboArmEnv_v0(
            max_episode_steps=env_conf.max_episode_steps,
            verbose=env_conf.verbose,
            sleep_time=env_conf.sleep_time,
            reward_signal=env_conf.reward_signal,
            pretrain=pretrain,
        )
    elif name == "roboarm_sim-v0":
        return RoboArmSimEnv_v0(
            max_episode_steps=env_conf.max_episode_steps,
            verbose=env_conf.verbose,
            noise=env_conf.noise,
            reward_signal=env_conf.reward_signal,
        )
    elif name == "roboarm_mixed-v0":
        return RoboArmMixedEnv_v0(
            max_episode_steps=env_conf.max_episode_steps,
            sleep_time=env_conf.sleep_time,
            verbose=env_conf.verbose,
            reward_signal=env_conf.reward_signal,
            camera_id=env_conf.camera_id,
            goal_radius=env_conf.goal_radius,
            pretrain=pretrain,
        )
    else:
        print("Environment not found")
