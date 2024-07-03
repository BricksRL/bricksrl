import os
import sys
import time

import hydra
import numpy as np
import wandb
from omegaconf import DictConfig, OmegaConf
from torchrl.envs.utils import step_mdp
from tqdm import tqdm

# Add the project root to PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from environments import make_env
from src.agents import get_agent
from src.utils import login, logout, prefill_buffer, setup_check, tensordict2dict


@hydra.main(version_base=None, config_path=project_root + "/conf", config_name="config")
def run(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # make environment.
    setup_check(robot="2wheeler", config=cfg)
    env, action_space, state_space = make_env(cfg)

    # make agent
    agent, project_name = get_agent(action_space, state_space, cfg)
    login(agent)

    # initialize wandb
    wandb.init(project=project_name)
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.watch(agent.actor, log_freq=1) if agent.actor else None

    # prefill buffer with random actions
    prefill_buffer(
        env=env,
        agent=agent,
        num_episodes=cfg.agent.prefill_episodes,
        stop_on_done=True if cfg.env.name == "runaway-v0" else False,
    )

    batch_size = cfg.agent.batch_size
    num_updates = cfg.agent.num_updates
    train_episodes = cfg.episodes
    print("Start training...")
    quit = False
    try:
        for e in tqdm(range(train_episodes), desc="Training"):
            td = env.reset()
            done = td.get("done", False)
            truncated = td.get("truncated", False)
            ep_return = 0
            ep_steps = 0
            total_step_times = []
            actions = []
            print("Start new data collection...", flush=True)
            while not done and not truncated:
                ep_steps += 1
                step_start_time = time.time()
                td = agent.get_action(td)
                actions.append(td.get("action").cpu().numpy())
                td = env.step(td)
                agent.add_experience(td)
                total_agent_step_time = time.time() - step_start_time
                total_step_times.append(total_agent_step_time)
                done = td.get(("next", "done"), False)
                ep_return += td.get(("next", "reward"), 0)
                if done:
                    if cfg.env.name == "runaway-v0":
                        inpt = input(
                            "Please reset the robot to the starting position and press Enter to continue or q to quit:"
                        )
                        if inpt == "q":
                            quit = True
                    break
                td = step_mdp(td)
            loss_info = agent.train(
                batch_size=batch_size, num_updates=num_updates * ep_steps
            )
            if quit:
                break

            # Metrics Logging
            log_dict = {
                "epoch": e,
                "reward": ep_return,
                "steps": ep_steps,
                "total_step_time": np.mean(total_step_times),
                "buffer_size": agent.replay_buffer.__len__(),
                "done": done.float(),
                "mean_action": np.mean(actions),
            }
            if cfg.env.name == "runaway-v0":
                log_dict.update({"distance": td.get("distance")})

            log_dict.update(tensordict2dict(loss_info))
            wandb.log(log_dict)

    except KeyboardInterrupt:
        print("Training interrupted by user.")

    logout(agent)
    env.close()


if __name__ == "__main__":
    run()
