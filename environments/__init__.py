from environments.donothing.donothing import DoNothingEnv
from environments.runaway_v0.RunAwayEnv import RunAwayEnv_v0
from environments.runaway_v1.RunAwayEnv import RunAwayEnv_v1
from environments.spinning_v1.SpinningEnv import SpinningEnv_v1
from environments.balance_v0.BalanceEnv import BalanceEnv_v0

from environments.walker_v0.WalkerEnv import WalkerEnv_v0
from environments.walker_v1.WalkerEnv import WalkerEnv_v1
from environments.walker_v2.WalkerEnv import WalkerEnv_v2
from environments.walkerwall_v0.WalkerWall import WalkerWallEnv_v0

from environments.roboarm_v0.RoboArmEnv import RoboArmEnv_v0
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
    CatFrames,)


def make_env(config):
    """
    Creates a new environment based on the provided configuration.

    Args:
        config: A configuration object containing the environment name and maximum episode steps.

    Returns:
        A tuple containing the new environment, its action space, and its state space.
    """
    env = make(name=config.env.name, env_conf=config.env)
    obs_key = env.observation_key

    transforms = []
    if config.env.frame_stack > 1:
        transforms.append(CatFrames(N=config.env.frame_stack, in_keys=[obs_key], out_key=obs_key))
    if config.env.action_filter < 1:
        raise NotImplementedError("ActionFilterWrapper not implemented yet")
        # TODO: add this to torchrl
        # env = ActionFilterWrapper(
        #     env, current_action_influence=config.env.action_filter
        # )
    env = TransformedEnv(env, Compose(*transforms))

    action_spec = env.action_spec
    state_spec = env.observation_spec

    return env, action_spec, state_spec
