import pytest
from environments.dummy.mixed_obs_dummy import MixedObsDummyEnv
from environments.dummy.vec_obs_dummy import VecObsDummyEnv
from hydra import compose, initialize
from src.agents import RandomAgent, SACAgent, TD3Agent
from torchrl.envs import Compose, ToTensorImage, TransformedEnv
from torchrl.envs.utils import step_mdp


def rollout(env, agent, max_steps=1000):
    td = env.reset()
    for _ in range(max_steps):
        print(td)
        td = agent.get_action(td)
        td = env.step(td)
        agent.add_experience(td)
        td = step_mdp(td)


def get_env(env):
    if env == "mixed":
        env = MixedObsDummyEnv()
        env = TransformedEnv(
            env, Compose(ToTensorImage(in_keys=["image_observation"], from_int=True))
        )
    else:
        env = VecObsDummyEnv()
    return env


@pytest.mark.parametrize(
    "env",
    ["mixed", "vec"],
)
def test_random_agent(env):
    with initialize(config_path="../conf"):
        cfg = compose(config_name="config")

    # Test data collection
    env = get_env(env)
    agent = RandomAgent(env.observation_spec, env.action_spec, cfg.agent)
    rollout(env, agent, max_steps=10)


@pytest.mark.parametrize(
    "env",
    ["mixed", "vec"],
)
def test_sac_agent(env):
    with initialize(config_path="../conf"):
        cfg = compose(config_name="config", overrides=["agent=sac"])

    # Test data collection
    env = get_env(env)
    agent = SACAgent(env.observation_spec, env.action_spec, cfg.agent)
    print(agent)
    rollout(env, agent, max_steps=10)
    # Test training
    agent.train(batch_size=1, num_updates=1)


@pytest.mark.parametrize(
    "env",
    ["mixed", "vec"],
)
def test_td3_agent(env):
    with initialize(config_path="../conf"):
        cfg = compose(config_name="config", overrides=["agent=td3"])

    # Test data collection
    env = get_env(env)
    agent = TD3Agent(env.observation_spec, env.action_spec, cfg.agent)
    rollout(env, agent, max_steps=10)
    # Test training
    agent.train(batch_size=1, num_updates=1)


@pytest.mark.parametrize(
    "env",
    ["mixed", "vec"],
)
def test_drq_agent(env):
    with initialize(config_path="../conf"):
        cfg = compose(config_name="config", overrides=["agent=drq"])

    # Test data collection
    env = get_env(env)
    agent = SACAgent(env.observation_spec, env.action_spec, cfg.agent)
    rollout(env, agent, max_steps=10)
    # Test training
    agent.train(batch_size=1, num_updates=1)
