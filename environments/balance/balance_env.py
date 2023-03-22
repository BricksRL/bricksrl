import torch
import numpy as np
from typing import Tuple

from torchrl.data import BoundedTensorSpec

from environments.base.base_env import BaseEnv
import time

class BalanceEnv(BaseEnv):
    def __init__(self,
                 max_episode_steps: int=10):
        
        action_dim = 1
        state_dim = 4 # 3 sensors (left, right, pitch,roll)
        self.normalize_factor = 80. # -80 / 80 range of pitch and roll # TODO adopt this for left right
        
        self.max_episode_steps = max_episode_steps
        
        self.action_space = BoundedTensorSpec(minimum=-torch.ones(action_dim),
                                              maximum=torch.ones(action_dim),
                                              shape=(action_dim,))
        self.observation_space = BoundedTensorSpec(minimum=torch.zeros(state_dim),
                                                   maximum=torch.ones(state_dim),
                                                   shape=(state_dim,))
        
        super().__init__(action_dim=action_dim, state_dim=state_dim)
    

    def sample_random_action(self)-> np.ndarray:
        """ Sample random action from action space. """
        action = np.random.uniform(self.action_space.minimum,
                                   self.action_space.maximum,
                                   size=self.action_dim)
        return action
    
    def normalize_state(self, state: np.ndarray)-> np.ndarray:
        """ Normalize observation """
        state[:, :2] /= 1000
        state[:, 2:] /= self.normalize_factor
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
        target_roll = 0.0
        # mse loss between roll and target roll
        # reward = - (next_state[:, 1] - target_roll)**2
        # calculate absolute error between roll and target roll
        #print("current roll", state[:, -1]* self.normalize_factor)
        #print("next roll", next_state[:, -1]* self.normalize_factor)
        #print("reward s", - np.abs(state[:, -1]* self.normalize_factor - target_roll))
        #print("reward ns", - np.abs(next_state[:, -1]* self.normalize_factor - target_roll))
        reward = - np.abs(next_state[:, -1]* self.normalize_factor - target_roll)         
        # moving penalty
        # move_penalty = (- (next_state[:, :2] - state[:, :2])**2).mean()
        move_penalty = - np.abs(next_state[:, :2] - state[:, :2]).mean() * 1000
        reward += 0.5 * move_penalty
        
        # include drive distance to reward as agent learns to drive forward or backward but keeps roll angle at 0

        return reward.item(), done
    
    def step(self, action: np.ndarray)-> Tuple[np.ndarray, float, bool, dict]:
        """ Perform action and return next state, reward and done. """

        # Send action to hub to receive next state
        self.send_to_hub(action)
        # time.sleep(0.2) # we need to wait some time for sensors to read and to 
                        # receive the next state
        next_obs = self.read_from_hub()
        print("Next State", next_obs)
        next_obs = self.normalize_state(next_obs)
        
        # calc reward and done
        reward, done = self.reward(state=self.observation, action=action, next_state=next_obs)
        
        print("Reward", reward)
        # set next state as current state
        self.observation = next_obs
        
        # increment episode step counter
        self.episode_step_iter += 1
        if self.episode_step_iter >= self.max_episode_steps:
            done = True
        
        return self.observation, reward, done, {}
    
