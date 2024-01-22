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
            state = env.reset()        
            done = False
            truncated = False
            ep_return = 0
            ep_steps = 0
            total_step_times = []
            agent_actions = []
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
                action = agent.get_action(state)
                if verbose == 1:
                    print("New step")
                    print("State: ", state)
                    print("Action: ", action)
                next_state, reward, done, truncated, info = env.step(action)
                transition = create_transition_td(state, action, np.array([reward]), next_state, np.array([done]))
                agent.add_experience(transition)
                state = next_state
                ep_return += reward

                total_agent_step_time = time.time() - step_start_time
                total_step_times.append(total_agent_step_time)
                
                if done:
                    inpt = input("Please reset the robot to the starting position and press Enter to continue or q to quit:")
                    if inpt == "q":
                       quit = True
                       break
            loss_info = agent.train(batch_size=batch_size,
                        num_updates=num_updates*ep_steps)
            agent_actions.append(action)
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
                        "action": wandb.Histogram(action),
                        "done": done,
                        "action_mean": wandb.Histogram(np.mean(agent_actions, axis=0))}
            log_dict.update(info)
            log_dict.update(tensordict2dict(loss_info))
            wandb.log(log_dict)
            
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    
    logout(agent)
    env.close()

if __name__ == "__main__":
    run()