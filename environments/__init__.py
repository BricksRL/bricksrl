from environments.donothing.donothing import DoNothingEnv
from environments.runaway_v0.RunAwayEnv import RunAwayEnv_v0
from environments.runaway_v1.RunAwayEnv import RunAwayEnv_v1
from environments.spinning_v1.SpinningEnv import SpinningEnv_v1
from environments.balance_v0.BalanceEnv import BalanceEnv_v0

from environments.walker_v0.WalkerEnv import WalkerEnv_v0
from environments.walker_v1.WalkerEnv import WalkerEnv_v1
from environments.walker_v2.WalkerEnv import WalkerEnv_v2
from environments.walkerwall_v0.WalkerWall import WalkerWallEnv_v0
from environments.walker_mixed_v0.WalkerEnv import WalkerMixedEnv_v0

from environments.roboarm_v0.RoboArmEnv import RoboArmEnv_v0
from environments.roboarm_mixed_v0.RoboArmMixedEnv import RoboArmMixedEnv_v0
from environments.utils import make

from environments.wrapper import (
    ActionFilterWrapper,
    FrameSkipWrapper,
    StackObservationsWrapper,
    StartControlWrapper,
    TorchEnvWrapper,
)
from torchrl.envs import (
    TransformedEnv,
    Compose,
    RewardSum,
    DoubleToFloat,
    CatFrames,
    ToTensorImage)

VIDEO_LOGGING_ENVS = ["roboar_mixed-v0", "walker_mixed-v0"]
ALL_2WHEELER_ENVS = ["Spinning-v1", "runaway-v0", "runaway-v1", "balance-v0"]
ALL_WALKER_ENVS = ["walker-v0", "walker-v1", "walker-v2", "WalkerWall-v0", "walker_mixed-v0"]
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
        transforms.append(CatFrames(N=config.env.frame_stack, in_keys=observation_keys, out_key=observation_keys))
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
