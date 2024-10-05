import os
import sys

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# Add the project root to PYTHONPATH for config
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from bricksrl.environments import make_env
from experiments.helper.agents import get_agent
from experiments.helper.utils import login, logout, setup_check, tensordict2dict


@hydra.main(version_base=None, config_path=project_root + "/conf", config_name="config")
def run(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # make environment.
    setup_check(robot="walker", config=cfg)
    env, action_space, state_space = make_env(cfg, pretrain=True)

    # make agent
    agent, project_name = get_agent(action_space, state_space, cfg)
    login(agent)

    # initialize wandb
    wandb.init(project=project_name)
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.watch(agent.actor, log_freq=1) if agent.actor else None

    batch_size = cfg.agent.batch_size
    num_updates = cfg.agent.num_updates
    train_episodes = cfg.episodes
    print("Start training...")
    try:
        for e in tqdm(range(train_episodes), desc="Training"):

            loss_info = agent.train(batch_size=batch_size, num_updates=num_updates)

            # Metrics Logging
            log_dict = {
                "epoch": e,
                "buffer_size": agent.replay_buffer.__len__(),
            }
            log_dict.update(tensordict2dict(loss_info))
            wandb.log(log_dict)

    except KeyboardInterrupt:
        print("Training interrupted by user.")

    logout(agent)
    env.close()


if __name__ == "__main__":
    run()
