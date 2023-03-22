import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import wandb
import time

from agents import TD3Agent, SACAgent
from environments import make
from environments.wrapper import TorchEnvWrapper, StartControlWrapper
from seb_examples.utils import logout, login, create_transition_td, tensordict2dict




@hydra.main(version_base=None, config_path="../conf", config_name="config")
def run(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    
    # Create environment.
    env = make(name=cfg.env_name, max_episode_steps=cfg.max_episode_steps)

    action_space = env.action_space
    state_space = env.observation_space
    
    # TODO: make create agent function
    # agent = TD3Agent(action_space=action_space, state_space=state_space, learning_rate=cfg.agent.lr, device=cfg.device)
    agent = SACAgent(action_space=action_space, state_space=state_space, learning_rate=cfg.agent.lr, device=cfg.device)
    
    login(agent)
    
    print("--- Agent initialized ---", flush=True)
    # Initialize wandb
    wandb.init(project="lego-balance-td3", config=cfg)
    wandb.watch(agent.actor, log_freq=1)
                             
    def prefill_buffer(env, agent, num_episodes):
        for e in range(num_episodes):
            inpt = input("Press Enter to start prefilling episode: ")
            print("Prefill episode: ", e)
            state = env.reset()
            done = False
            while not done:
                action = np.random.uniform(-1, 1, size=1)
                print("Random action: ", action)
                next_state, reward, done, info = env.step(action)
                transition = create_transition_td(state, action, np.array([reward]), next_state, np.array([done]))
                agent.add_experience(transition)
                state = next_state
    
    prefill_buffer(env, agent, 3)
    print("Prefill done! Buffer size: ", agent.replay_buffer.__len__())
    
    print("Start training...")
    quit = False
    for e in range(cfg.episodes):
        state = env.reset()        
        done = False
        ep_return = 0
        # Train agent
        # loss_info = agent.train(batch_size=cfg.agent.batch_size,
        #                        num_updates=cfg.agent.num_updates)

        print("Start new data collection...", flush=True)
        print("Init done: ", done)
        inpt = input("Press Enter to start episode or q to quit: ")
        if inpt == "q":
            done = True
            quit = True
        while not done:
            step_start_time = time.time()
            action = agent.get_action(state)
            print("New step")
            print("State: ", state)
            print("Action: ", action)
            next_state, reward, done, info = env.step(action)
            print("Done: ", done)
            transition = create_transition_td(state, action, np.array([reward]), next_state, np.array([done]))
            agent.add_experience(transition)
            state = next_state
            ep_return += reward
            loss_info = agent.train(batch_size=cfg.agent.batch_size,
                        num_updates=cfg.agent.num_updates)
            print("Step time: ", time.time() - step_start_time)
            print("---"*5)
        if quit:
            break

        print("Episode: ", e, "Return: ", ep_return)
        # Metrics Logging
        log_dict = {"epoch": e,
                    "reward": ep_return,
                    "buffer_size": agent.replay_buffer.__len__(),
                    "final_distance": state[:, -1]}
        log_dict.update(tensordict2dict(loss_info))
        wandb.log(log_dict)
        
    
    logout(agent)
    env.close()

if __name__ == "__main__":
    run()