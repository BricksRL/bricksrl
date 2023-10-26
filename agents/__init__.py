from agents.sac import SACAgent
from agents.td3 import TD3Agent

all_agents = ["td3", "sac"]


def get_agent(action_space, state_space, cfg):
    if cfg.agent.name == "td3":
        agent = TD3Agent(
            action_space=action_space,
            state_space=state_space,
            agent_config=cfg.agent,
            device=cfg.device,
        )
    elif cfg.agent.name == "sac":
        agent = SACAgent(
            action_space=action_space,
            state_space=state_space,
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
