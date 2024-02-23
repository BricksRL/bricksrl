import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import wandb
import time
import sys
sys.path.insert(0, '/home/sebastian/Documents/lego_robot')

from agents import get_agent
from utils import prefill_buffer, logout, login, tensordict2dict, create_video_from_images
from environments import make_env


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def run(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    verbose = cfg.verbose
    checking_mode = cfg.checking_mode
    # make environment.
    env, action_space, state_space = make_env(cfg)
    
    # make agent
    agent, project_name = get_agent(action_space, state_space, cfg)
    login(agent)
    
    # initialize wandb
    wandb.init(project=project_name)
    wandb.config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    wandb.watch(agent.actor, log_freq=1) if agent.actor else None

    # prefill buffer  
    prefill_buffer(env=env, agent=agent, checking_mode=checking_mode, num_episodes=cfg.agent.prefill_episodes)

    # pretraining
    if agent.do_pretrain:
        print("Start pretraining...")
        agent.pretrain(wandb, batch_size=cfg.agent.batch_size, num_updates=cfg.agent.num_updates)
        print("Pretraining finished.")
        inp = input("Press Enter to start online training: ")


    batch_size = cfg.agent.batch_size
    num_updates = cfg.agent.num_updates
    print("Start training...")
    quit = False
    try:
        for e in range(cfg.episodes):
            td = env.reset()        
            done = td.get("done", False)
            truncated = td.get("truncated", False)
            ep_return = 0
            ep_steps = 0
            total_step_times = []
            # agent_actions = []
            image_caputres = []
            image_caputres.append(td.get("original_image").numpy())
            if checking_mode == 1:
                inp = input("Press Enter to start episode: ")
                if inp == "q":
                    quit = True
                    break
                else:
                    pass
            else:
                pass 
            print("Start new data collection...", flush=True)
            while not done and not truncated:
                ep_steps += 1
                step_start_time = time.time()
                td = agent.get_action(td)
                td = env.step(td)
                image_caputres.append(td.get("original_image").numpy())
                agent.add_experience(td)
                total_agent_step_time = time.time() - step_start_time
                total_step_times.append(total_agent_step_time)
                done = td.get(("next", "done"), False)   
                ep_return += td.get(("next","reward"), 0)             
                if done:
                    # inpt = input("Please reset the robot to the starting position and press Enter to continue or q to quit:")
                    # if inpt == "q":
                    #     quit = True
                    break
            loss_info = agent.train(batch_size=batch_size,
                        num_updates=num_updates*ep_steps)
            #agent_actions.append(action)
            if quit:
                break
          
            if verbose == 1:
                print("Episode: ", e, "Return: ", ep_return)
            # Metrics Logging
            log_dict = {"epoch": e,
                        "reward": ep_return,
                        "steps": ep_steps,
                        "total_step_time": np.mean(total_step_times),
                        "buffer_size": agent.replay_buffer.__len__(),
                        #"action": wandb.Histogram(action),
                        "done": done,}
                        # "action_mean": wandb.Histogram(np.mean(agent_actions, axis=0))}
            # log_dict.update(info)
            log_dict.update(tensordict2dict(loss_info))
            wandb.log(log_dict)
            video_name = "episode_{}".format(e)
            create_video_from_images(image_caputres, video_name)
            wandb.log({"video": wandb.Video(video_name, fps=20, format="mp4")})

    except KeyboardInterrupt:
        print("Training interrupted by user.")
    
    logout(agent)
    env.close()

if __name__ == "__main__":
    run()