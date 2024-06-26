import torch
from torchrl.envs import (
    CatFrames,
    Compose,
    DoubleToFloat,
    RewardSum,
    ToTensorImage,
    TransformedEnv,
)

from environments.roboarm_mixed_v0.RoboArmMixedEnv import RoboArmMixedEnv_v0
from environments.roboarm_pickplace_v0.RoboArmPickPlaceEnv import RoboArmPickPlaceEnv_v0
from environments.roboarm_v0.RoboArmEnv import RoboArmEnv_v0
from environments.roboarm_v0.RoboArmSim import RoboArmSimEnv_v0
from environments.runaway_v0.RunAwayEnv import RunAwayEnv_v0
from environments.spinning_v0.SpinningEnv import SpinningEnv_v0
from environments.walker_v0.WalkerEnv import WalkerEnv_v0
from environments.walker_v0.WalkerEnvSim import WalkerEnvSim_v0

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
    "roboarm_pickplace-v0",
]
ALL_ENVS = ALL_2WHEELER_ENVS + ALL_WALKER_ENVS + ALL_ROBOARM_ENVS


def make_env(config):
    """
    Creates a new environment based on the provided configuration.

    Args:
        config: A configuration object containing the environment name and maximum episode steps.

    Returns:
        A tuple containing the new environment, its action space, and its state space.
    """
    env = make(name=config.env.name, env_conf=config.env)
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
    if "image_observation" in observation_keys:
        transforms.append(
            ToTensorImage(
                in_keys=["image_observation"], from_int=False, dtype=torch.uint8
            )
        )  # from_int=True, dtype=torch.float32

    env = TransformedEnv(env, Compose(*transforms))

    action_spec = env.action_spec
    state_spec = env.observation_spec

    return env, action_spec, state_spec


def make(name="RunAway", env_conf=None):
    if name == "runaway-v0":
        return RunAwayEnv_v0(
            max_episode_steps=env_conf.max_episode_steps,
            min_distance=env_conf.min_distance,
            verbose=env_conf.verbose,
        )
    elif name == "spinning-v0":
        return SpinningEnv_v0(
            max_episode_steps=env_conf.max_episode_steps,
            sleep_time=env_conf.sleep_time,
            verbose=env_conf.verbose,
        )
    elif name == "walker-v0":
        return WalkerEnv_v0(
            max_episode_steps=env_conf.max_episode_steps,
            verbose=env_conf.verbose,
            sleep_time=env_conf.sleep_time,
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
        )
    elif name == "roboarm_pickplace-v0":
        return RoboArmPickPlaceEnv_v0(
            max_episode_steps=env_conf.max_episode_steps,
            sleep_time=env_conf.sleep_time,
            verbose=env_conf.verbose,
            reward_signal=env_conf.reward_signal,
            camera_id=env_conf.camera_id,
        )
    else:
        print("Environment not found")
