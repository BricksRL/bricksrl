import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import wandb
import time
import sys
sys.path.insert(0, '/home/sebastian/Documents/lego_robot')

from agents import get_agent
from utils import prefill_buffer, logout, login, create_transition_td, tensordict2dict
from environments import make_env

@hydra.main(version_base=None, config_path="./conf", config_name="config")
def run(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    verbose = cfg.verbose
    # make environment.
    env, action_space, state_space = make_env(cfg)
    
    # make agent
    agent, project_name = get_agent(action_space, state_space, cfg)
    login(agent)
    
    # initialize wandb
    wandb.init(project=project_name, config=cfg)
    wandb.watch(agent.actor, log_freq=1)

    # prefill buffer  
    prefill_buffer(env=env, agent=agent, num_episodes=cfg.agent.prefill_episodes)
    
    print("Start training...")
    quit = False
    for e in range(cfg.episodes):
        state = env.reset()        
        done = False
        ep_return = 0
        print("Start new data collection...", flush=True)
        print("Init done: ", done)
        inpt = input("Press Enter to start episode or q to quit: ")
        if inpt == "q":
            done = True
            quit = True
        while not done:
            step_start_time = time.time()
            action = agent.get_action(state)
            if verbose == 1:
                print("New step")
                print("State: ", state)
                print("Action: ", action)
            next_state, reward, done, info = env.step(action)
            transition = create_transition_td(state, action, np.array([reward]), next_state, np.array([done]))
            agent.add_experience(transition)
            state = next_state
            ep_return += reward
            loss_info = agent.train(batch_size=cfg.agent.batch_size,
                        num_updates=cfg.agent.num_updates)
            if verbose == 1:
                print("Done: ", done)
                print("Step time: ", time.time() - step_start_time)
                print("---"*5)
        if quit:
            break
        
        if verbose == 1:
            print("Episode: ", e, "Return: ", ep_return)
        # Metrics Logging
        log_dict = {"epoch": e,
                    "reward": ep_return,
                    "buffer_size": agent.replay_buffer.__len__(),
                    "final_distance": state[-1]}
        log_dict.update(tensordict2dict(loss_info))
        wandb.log(log_dict)
        
    
    logout(agent)
    env.close()

if __name__ == "__main__":
    run()