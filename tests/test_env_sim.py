import pytest
import torch
from bricksrl.environments import make_env
from experiments.helper.agents import get_agent
from hydra import compose, initialize

from tests.test_agents import collection_round


@pytest.mark.parametrize(
    "agent",
    ["sac", "td3", "random"],
)
@pytest.mark.parametrize("env", ["walker_sim-v0", "roboarm_sim-v0"])
@pytest.mark.parametrize(
    "device",
    ["cpu", "cuda"],
)
def test_sac_agent(agent, env, device):
    if torch.cuda.is_available() and device == "cuda":
        device = "cuda"
    else:
        device = "cpu"
    with initialize(config_path="../conf"):
        cfg = compose(
            config_name="config",
            overrides=["agent=" + agent, "device=" + device, "env=" + env],
        )

    # Create environment
    env, action_space, state_space = make_env(cfg)
    # Create agent
    agent, _ = get_agent(action_space, state_space, cfg)
    print(agent)
    # Test data collection
    collection_round(env, agent, max_steps=10)
