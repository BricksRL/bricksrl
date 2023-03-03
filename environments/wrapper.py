import torch
from typing import Tuple
import numpy as np
import time

class BaseWrapper(object):
    def __init__(self, env):
        super.__init__(env)
        self.env = env

    def step(self, action):
        """Runs the environment :meth:`env.step` using the modified ``action`` from :meth:`self.action`."""
        raise NotImplementedError

    def action(self, action):
        """Returns a modified action before :meth:`env.step` is called."""
        raise NotImplementedError

    def reverse_action(self, action):
        """Returns a reversed ``action``."""
        raise NotImplementedError

class TorchEnvWrapper(BaseWrapper):
    def __init__(self, env, device="cpu"):
        self.env = env
        self.device = device
        
    def reset(self)-> torch.Tensor:
        obs = self.env.reset()
        return torch.from_numpy(obs).float().to(self.device)
        
    def step(self, action: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        action = action.cpu().numpy()
        obs, reward, done, info = self.env.step(action)
        return (torch.from_numpy(obs).float().to(self.device),
                torch.from_numpy(np.array(reward)).float().to(self.device),
                torch.from_numpy(np.array(done)).float().to(self.device),
                info)
        
        
class StartControlWrapper(BaseWrapper):
    def __init__(self, env):
        self.env = env

    def reset(self)-> torch.Tensor:
        time.sleep(1)
        print("\nSet the robot in the starting position.")
        time.sleep(1)
        start_verifier = input("Press enter to start the environment.")
        return self.env.reset()
    
    def step(self, action):
        return self.env.step(action)