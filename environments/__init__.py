from torchrl.envs import (
    CatFrames,
    Compose,
    DoubleToFloat,
    RewardSum,
    ToTensorImage,
    TransformedEnv,
)

from environments.balance_v0.BalanceEnv import BalanceEnv_v0
from environments.roboarm_mixed_v0.RoboArmMixedEnv import RoboArmMixedEnv_v0
from environments.roboarm_v0.RoboArmEnv import RoboArmEnv_v0
from environments.runaway_v0.RunAwayEnv import RunAwayEnv_v0
from environments.runaway_v1.RunAwayEnv import RunAwayEnv_v1
from environments.spinning_v0.SpinningEnv import SpinningEnv_v0
from environments.walker_mixed_v0.WalkerEnv import WalkerMixedEnv_v0
from environments.walker_v0.WalkerEnv import WalkerEnv_v0
from environments.walker_v1.WalkerEnv import WalkerEnv_v1
from environments.walker_v2.WalkerEnv import WalkerEnv_v2


VIDEO_LOGGING_ENVS = ["roboar_mixed-v0", "walker_mixed-v0", "walker_mixed_v1"]
ALL_2WHEELER_ENVS = ["spinning-v0", "runaway-v0", "runaway-v1", "balance-v0"]
ALL_WALKER_ENVS = [
    "walker-v0",
    "walker-v1",
    "walker-v2",
    "walker_mixed-v0",
    "walker_mixed-v1",
]
ALL_ROBOARM_ENVS = ["roboarm-v0", "roboarm_mixed-v0"]
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
        transforms.append(ToTensorImage(in_keys=["image_observation"], from_int=True))

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
    elif name == "runaway-v1":
        return RunAwayEnv_v1(
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
    elif name == "balance-v0":
        return BalanceEnv_v0(
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
    elif name == "walker-v1":
        return WalkerEnv_v1(
            max_episode_steps=env_conf.max_episode_steps,
            max_acc=env_conf.max_acceleration,
            reward_normalization_factor=env_conf.reward_normalization_factor,
            reward_clip=env_conf.reward_clip,
            verbose=env_conf.verbose,
            sleep_time=env_conf.sleep_time,
        )
    elif name == "walker-v2":
        return WalkerEnv_v2(
            max_episode_steps=env_conf.max_episode_steps,
            max_acc=env_conf.max_acceleration,
            max_distance=env_conf.max_distance,
            reward_normalization_factor=env_conf.reward_normalization_factor,
            reward_clip_acc=env_conf.reward_clip_acc,
            use_acceleration_reward=env_conf.use_acceleration_reward,
            verbose=env_conf.verbose,
            sleep_time=env_conf.sleep_time,
        )
    elif name == "walker_mixed-v0":
        return WalkerMixedEnv_v0(
            max_episode_steps=env_conf.max_episode_steps,
            sleep_time=env_conf.sleep_time,
            verbose=env_conf.verbose,
            camera_id=env_conf.camera_id,
            image_path=env_conf.image_path,
            tracker_type=env_conf.tracker_type,
            auto_obj_detection=env_conf.auto_obj_detection,
        )
    elif name == "walker_mixed-v1":
        return WalkerMixedEnv_v0(
            max_episode_steps=env_conf.max_episode_steps,
            sleep_time=env_conf.sleep_time,
            verbose=env_conf.verbose,
            camera_id=env_conf.camera_id,
            image_path=env_conf.image_path,
            tracker_type=env_conf.tracker_type,
            auto_obj_detection=env_conf.auto_obj_detection,
        )
    elif name == "roboarm-v0":
        return RoboArmEnv_v0(
            max_episode_steps=env_conf.max_episode_steps,
            verbose=env_conf.verbose,
            sleep_time=env_conf.sleep_time,
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
    else:
        print("Environment not found")
