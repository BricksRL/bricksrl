from environments.donothing.donothing import DoNothingEnv
from environments.runaway_v0.RunAwayEnv import RunAwayEnv_v0
from environments.runaway_v1.RunAwayEnv import RunAwayEnv_v1
from environments.walker_v0.WalkerEnv import WalkerEnv_v0
from environments.utils import make

from environments.wrapper import (
    ActionFilterWrapper,
    FrameSkipWrapper,
    StackObservationsWrapper,
    StartControlWrapper,
    TorchEnvWrapper,
)


def make_env(config):
    """
    Creates a new environment based on the provided configuration.

    Args:
        config: A configuration object containing the environment name and maximum episode steps.

    Returns:
        A tuple containing the new environment, its action space, and its state space.
    """
    env = make(name=config.env.name, env_conf=config.env)

    if config.env.frame_stack > 1:
        env = StackObservationsWrapper(env, stack_size=config.env.frame_stack)
    if config.env.action_filter < 1:
        env = ActionFilterWrapper(
            env, current_action_influence=config.env.action_filter
        )
    
    # env = FrameSkipWrapper(env, frame_skip=4)

    action_space = env.action_space
    state_space = env.observation_space

    return env, action_space, state_space
