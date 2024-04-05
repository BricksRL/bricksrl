# TODO
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

from environments import make_env, VIDEO_LOGGING_ENVS
from src.agents import get_agent
from src.utils import create_video_from_images, login, setup_check

"""
Goal positions for the evaluation:

    stretch-out: [GM: -130, RM: 0, LM: 70, HM: -130]
    grab-up: [GM: -130, RM: 0, LM: 70, HM: 25]
    hold-high: [GM: -50, RM: 0, LM: 15, HM: -130]
    random: [GM: -100, RM: 120, LM: 34, HM: -90]
"""


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
                image_caputres = [td.get("original_image").numpy()]
            if env_name == "roboarm-v0":
                goal_state = td.get(env.original_goal_observation_key).cpu().numpy()
            print("Start new evaluation...", flush=True)
            while not done and not truncated:
                ep_steps += 1
                step_start_time = time.time()
                td = agent.get_eval_action(td)
                print("***" * 10)
                print("Goal State: ", td.get("original_goal_state"))
                print("Normalized Goal State: ", td.get("goal_state"))
                print("State: ", td.get("original_state"))
                print("Normalized State: ", td.get("vec_observation"))
                print("Action: ", td.get("action"))
                td = env.step(td)
                if env_name in VIDEO_LOGGING_ENVS:
                    image_caputres.append(
                        td.get(("next", "original_image")).cpu().numpy()
                    )
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
            if env_name == "roboarm-v0":
                achieved_state = td.get(env.original_observation_key).cpu().numpy()
                error = np.linalg.norm(achieved_state - goal_state)
                print("---" * 10)
                print("Goal state: ", goal_state)
                print("Achieved state: ", achieved_state)
                print("Error: ", error)
                log_dict["final_error"] = error

            wandb.log(log_dict)
            if env_name in VIDEO_LOGGING_ENVS:
                video_name = "episode_{}.mp4".format(e)
                create_video_from_images(image_caputres, video_name, fps=5)
                wandb.log({"video": wandb.Video(video_name, fps=5, format="mp4")})

    except KeyboardInterrupt:
        print("Evaluation interrupted by user.")

    env.close()


if __name__ == "__main__":
    run()
