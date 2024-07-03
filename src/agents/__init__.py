from src.agents.behavior_cloning import BehavioralCloningAgent
from src.agents.cql import CQLAgent
from src.agents.iql import IQLAgent
from src.agents.keyboard import KeyboardAgent
from src.agents.random import RandomAgent
from src.agents.sac import SACAgent
from src.agents.td3 import TD3Agent

all_agents = ["td3", "sac", "iql", "cql", "random", "keyboard"]


def get_agent(action_spec, state_spec, cfg):
    if cfg.agent.name == "td3":
        agent = TD3Agent(
            action_spec=action_spec,
            state_spec=state_spec,
            agent_config=cfg.agent,
            device=cfg.device,
        )
    elif cfg.agent.name == "sac":
        agent = SACAgent(
            action_spec=action_spec,
            state_spec=state_spec,
            agent_config=cfg.agent,
            device=cfg.device,
        )
    elif cfg.agent.name == "bc":
        agent = BehavioralCloningAgent(
            action_spec=action_spec,
            state_spec=state_spec,
            agent_config=cfg.agent,
            device=cfg.device,
        )
    elif cfg.agent.name == "random":
        agent = RandomAgent(
            action_spec=action_spec,
            state_spec=state_spec,
            agent_config=cfg.agent,
            device=cfg.device,
        )
    elif cfg.agent.name == "iql":
        agent = IQLAgent(
            action_spec=action_spec,
            state_spec=state_spec,
            agent_config=cfg.agent,
            device=cfg.device,
        )
    elif cfg.agent.name == "cql":
        agent = CQLAgent(
            action_spec=action_spec,
            state_spec=state_spec,
            agent_config=cfg.agent,
            device=cfg.device,
        )
    elif cfg.agent.name == "keyboard":
        agent = KeyboardAgent(
            action_spec=action_spec,
            state_spec=state_spec,
            agent_config=cfg.agent,
            device=cfg.device,
        )
    else:
        raise NotImplementedError(
            f"Agent {cfg.agent.name} not implemented, please choose from {all_agents}"
        )

    project_name = f"lego-{cfg.agent.name}-{cfg.env.name}"
    print("--- Agent initialized ---", flush=True)

    return agent, project_name
