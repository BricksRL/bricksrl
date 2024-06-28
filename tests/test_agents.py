import pytest
import torch
from environments.dummy.mixed_obs_dummy import MixedObsDummyEnv
from environments.dummy.vec_obs_dummy import VecGoalObsDummyEnv, VecObsDummyEnv
from hydra import compose, initialize
from src.agents import get_agent
from torchrl.envs import Compose, ToTensorImage, TransformedEnv
from torchrl.envs.utils import step_mdp


def collection_round(env, agent, max_steps=1000):
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
            env, Compose(ToTensorImage(in_keys=["pixels"], from_int=True))
        )
    elif env == "vec":
        env = VecObsDummyEnv()
    elif env == "vec_goal":
        env = VecGoalObsDummyEnv()
    else:
        raise ValueError("Invalid environment")
    return env


@pytest.mark.parametrize(
    "env",
    ["mixed", "vec", "vec_goal"],
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
    agent, _ = get_agent(env.action_spec, env.observation_spec, cfg)
    collection_round(env, agent, max_steps=10)


@pytest.mark.parametrize(
    "env",
    ["mixed", "vec", "vec_goal"],
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
    agent, _ = get_agent(env.action_spec, env.observation_spec, cfg)
    collection_round(env, agent, max_steps=10)
    # Test training
    agent.train(batch_size=1, num_updates=1)

    # Test evaluation
    td = env.reset()
    td1 = agent.get_action(td)
    td2 = agent.get_action(td)

    assert not torch.allclose(td1["action"], td2["action"])

    agent.eval()
    td = env.reset()
    eval_td1 = agent.get_eval_action(td)
    eval_td2 = agent.get_eval_action(td)

    assert torch.allclose(eval_td1["action"], eval_td2["action"])


@pytest.mark.parametrize(
    "env",
    ["mixed", "vec", "vec_goal"],
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
    agent, _ = get_agent(env.action_spec, env.observation_spec, cfg)
    collection_round(env, agent, max_steps=10)

    # Test training
    agent.train(batch_size=1, num_updates=1)

    # Test evaluation
    td = env.reset()
    td1 = agent.get_action(td)
    td2 = agent.get_action(td)

    assert not torch.allclose(td1["action"], td2["action"])

    agent.eval()
    td = env.reset()
    eval_td1 = agent.get_eval_action(td)
    eval_td2 = agent.get_eval_action(td)

    assert torch.allclose(eval_td1["action"], eval_td2["action"])


@pytest.mark.parametrize(
    "env",
    ["mixed", "vec", "vec_goal"],
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
        cfg = compose(
            config_name="config", overrides=["agent=droq", "device=" + device]
        )

    # Test data collection
    env = get_env(env)
    agent, _ = get_agent(env.action_spec, env.observation_spec, cfg)
    collection_round(env, agent, max_steps=10)
    # Test training
    agent.train(batch_size=1, num_updates=1)

    # Test evaluation
    td = env.reset()
    td1 = agent.get_action(td)
    td2 = agent.get_action(td)

    assert not torch.allclose(td1["action"], td2["action"])

    agent.eval()
    td = env.reset()
    eval_td1 = agent.get_eval_action(td)
    eval_td2 = agent.get_eval_action(td)

    assert torch.allclose(eval_td1["action"], eval_td2["action"])
