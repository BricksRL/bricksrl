import pytest
import torch
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
@pytest.mark.parametrize(
    "device",
    ["cpu", "cuda"],
)
def test_random_agent(env, device):
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
@pytest.mark.parametrize(
    "device",
    ["cpu", "cuda"],
)
def test_sac_agent(env, device):
    if torch.cuda.is_available() and device == "cuda":
        device = "cuda"
    else:
        device = "cpu"
    with initialize(config_path="../conf"):
        cfg = compose(config_name="config", overrides=["agent=sac", "device=" + device])

    # Test data collection
    env = get_env(env)
    agent = SACAgent(env.observation_spec, env.action_spec, cfg.agent, device=device)
    print(agent)
    rollout(env, agent, max_steps=10)
    # Test training
    agent.train(batch_size=1, num_updates=1)


@pytest.mark.parametrize(
    "env",
    ["mixed", "vec"],
)
@pytest.mark.parametrize(
    "device",
    ["cpu", "cuda"],
)
def test_td3_agent(env, device):
    if torch.cuda.is_available() and device == "cuda":
        device = "cuda"
    else:
        device = "cpu"
    with initialize(config_path="../conf"):
        cfg = compose(config_name="config", overrides=["agent=td3", "device=" + device])

    # Test data collection
    env = get_env(env)
    agent = TD3Agent(env.observation_spec, env.action_spec, cfg.agent, device=device)
    rollout(env, agent, max_steps=10)
    # Test training
    agent.train(batch_size=1, num_updates=1)


@pytest.mark.parametrize(
    "env",
    ["mixed", "vec"],
)
@pytest.mark.parametrize(
    "device",
    ["cpu", "cuda"],
)
def test_drq_agent(env, device):
    if torch.cuda.is_available() and device == "cuda":
        device = "cuda"
    else:
        device = "cpu"
    with initialize(config_path="../conf"):
        cfg = compose(config_name="config", overrides=["agent=drq", "device=" + device])

    # Test data collection
    env = get_env(env)
    agent = SACAgent(env.observation_spec, env.action_spec, cfg.agent, device=device)
    rollout(env, agent, max_steps=10)
    # Test training
    agent.train(batch_size=1, num_updates=1)
