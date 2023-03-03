import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np

from agents import TD3Agent, SACAgent
from environments import RunAwayEnv
from environments.wrapper import TorchEnvWrapper, StartControlWrapper

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def run(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    
    # Create environment.
    env = StartControlWrapper(RunAwayEnv(max_episode_steps=10,
                                     max_distance=1000.,
                                     min_distance=40.)) #(TorchEnvWrapper(
    state = env.reset()
    print("state", state)

    for e in range(cfg.episodes):
        done = False
        while not done:
            action = np.random.uniform(-1, 1, size=1)
            next_state, reward, done, info = env.step(action)
            print("next_state", next_state)
            print("reward", reward)
            print("done", done)
            print("info", info)
            print("action", action)

    




if __name__ == "__main__":
    run()