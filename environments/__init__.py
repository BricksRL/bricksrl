from environments.donothing.donothing import DoNothingEnv
from environments.runaway.RunAwayEnv import RunAwayEnv
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
    env = make(name=config.env.name, max_episode_steps=config.env.max_episode_steps)

    env = StackObservationsWrapper(env, stack_size=4)
    # env = FrameSkipWrapper(env, frame_skip=4)
    env = ActionFilterWrapper(env, current_action_influence=0.8)

    action_space = env.action_space
    state_space = env.observation_space

    return env, action_space, state_space
