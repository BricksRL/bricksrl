import torch
import numpy as np
from typing import Tuple

from torchrl.data import BoundedTensorSpec

from environments.base.base_env import BaseEnv
import time

class RunAwayEnv(BaseEnv):
    def __init__(self,
                 max_episode_steps: int=10,
                 max_distance: float = 1000.,
                 min_distance: float = 40):
        
        action_dim = 1
        state_dim = 5 # 4 sensors (left,right,pitch,roll) + 1 distance to the wall
        self.normalize_factor = 1000.
        self.max_distance = max_distance
        self.min_distance = min_distance / self.normalize_factor
        
        self.max_episode_steps = max_episode_steps
        
        self.action_space = BoundedTensorSpec(minimum=-torch.ones(action_dim),
                                              maximum=torch.ones(action_dim),
                                              shape=(action_dim,))
        self.observation_space = BoundedTensorSpec(minimum=torch.zeros(state_dim),
                                              maximum=torch.ones(state_dim)*self.max_distance,
                                              shape=(state_dim,))
        
        super().__init__(action_dim=action_dim, state_dim=state_dim)
    

    def sample_random_action(self)-> np.ndarray:
        """ Sample random action from action space. """
        action = np.random.uniform(self.action_space.minimum,
                                   self.action_space.maximum,
                                   size=self.action_dim)
        return action
    
    def normalize_state(self, state: np.ndarray)-> np.ndarray:
        """ Normalize and clips state to be compatible with the agent. """
        state = np.clip(state, 0, self.max_distance)
        state = state / self.normalize_factor
        return state

    def reset(self)-> np.ndarray:
        """ Reset environment and return initial state. """
        # TODO solve this fake action sending before to receive first state
        self.episode_step_iter = 0

        self.send_to_hub(np.array([0.001]))
        time.sleep(0.4)
        self.observation = self.normalize_state(self.read_from_hub())

        return self.observation
    
    def reward(self,
               state: np.ndarray,
               action: np.ndarray,
               next_state: np.ndarray)-> Tuple[float, bool]:
        """ Reward function of RunAwayEnv.

            Goal: Increase distance measured by ultrasonic sensor aka.
            get away from the wall as fast as possible.
        
        """
        done = False
        
        if next_state[:, -1] <= self.min_distance: # too close to the wall break episode
            reward = -10
            done = True
        elif next_state[:, -1] < state[:, -1]:
            reward = -1.
        elif next_state[:, -1] > state[:, -1]:
            reward = 1.
        else:
            reward = 0.

        return reward, done
    
    def step(self, action: np.ndarray)-> Tuple[np.ndarray, float, bool, dict]:
        """ Perform action and return next state, reward and done. """

        # Send action to hub to receive next state
        self.send_to_hub(action)
        time.sleep(0.2) # we need to wait some time for sensors to read and to 
                        # receive the next state
        next_observation = self.normalize_state(self.read_from_hub())
        
        # calc reward and done
        reward, done = self.reward(state=self.observation, action=action, next_state=next_observation)
        
        # set next state as current state
        self.observation = next_observation
        
        # increment episode step counter
        self.episode_step_iter += 1
        if self.episode_step_iter >= self.max_episode_steps:
            done = True
        
        return self.observation, reward, done, {}
    
