import os
import sys
import time

import hydra
import numpy as np
import wandb
from omegaconf import DictConfig, OmegaConf
from torchrl.envs.utils import step_mdp
from tqdm import tqdm

# Add the project root to PYTHONPATH for config
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from bricksrl.environments import make_env, VIDEO_LOGGING_ENVS
from experiments.helper.agents import get_agent
from experiments.helper.utils import (
    create_video_from_images,
    login,
    logout,
    setup_check,
)


@hydra.main(version_base=None, config_path=project_root + "/conf", config_name="config")
def run(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # make environment.
    setup_check(robot="roboarm", config=cfg)
    env, action_space, state_space = make_env(cfg)

    # make agent
    agent, project_name = get_agent(action_space, state_space, cfg)
    login(agent)
    agent.eval()

    # initialize wandb
    wandb.init(project=project_name + "_eval")
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    eval_episodes = cfg.episodes
    env_name = cfg.env.name
    quit = False
    _ = input("Press Enter to start evaluation...")
    try:
        for e in tqdm(range(eval_episodes), desc="Evaluation"):
            td = env.reset()
            done = td.get("done", False)
            truncated = td.get("truncated", False)
            ep_return = 0
            ep_steps = 0
            total_step_times = []
            if env_name in VIDEO_LOGGING_ENVS:
                image_caputres = [td.get("original_pixels").numpy()]
            print("Start new evaluation...", flush=True)
            while not done and not truncated:
                ep_steps += 1
                step_start_time = time.time()
                td = agent.get_eval_action(td)
                td = env.step(td)
                agent.add_experience(td)
                if env_name in VIDEO_LOGGING_ENVS:
                    image_caputres.append(
                        td.get(("next", "original_pixels")).cpu().numpy()
                    )
                agent.add_experience(td)
                total_agent_step_time = time.time() - step_start_time
                total_step_times.append(total_agent_step_time)
                done = td.get(("next", "done"), False)
                ep_return += td.get(("next", "reward"), 0)

                if done:
                    break
                td = step_mdp(td)

            if quit:
                break

            # Metrics Logging
            log_dict = {
                "epoch": e,
                "reward": ep_return,
                "steps": ep_steps,
                "total_step_time": np.mean(total_step_times),
                "done": done.float(),
            }
            if env_name == "roboarm-v0" or env_name == "roboarm_sim-v0":
                final_error = td.get(("error")).item()
                log_dict.update({"final_error": final_error})

            wandb.log(log_dict)
            if env_name in VIDEO_LOGGING_ENVS:
                video_name = "episode_{}.mp4".format(e)
                create_video_from_images(image_caputres, video_name, fps=5)
                wandb.log({"video": wandb.Video(video_name, fps=5, format="mp4")})

    except KeyboardInterrupt:
        print("Evaluation interrupted by user.")

    logout(agent)
    env.close()


if __name__ == "__main__":
    run()
