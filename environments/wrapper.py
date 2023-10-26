import time
from typing import Tuple

import gym
import numpy as np
import torch
from gym import Env


class BaseWrapper(gym.Wrapper):
    def __init__(self, env):
        super.__init__(env)
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

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

    def reset(self) -> torch.Tensor:
        obs = self.env.reset()
        return torch.from_numpy(obs).float().to(self.device)

    def step(
        self, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        action = action.cpu().numpy()
        obs, reward, done, info = self.env.step(action)
        return (
            torch.from_numpy(obs).float().to(self.device),
            torch.from_numpy(np.array(reward)).float().to(self.device),
            torch.from_numpy(np.array(done)).float().to(self.device),
            info,
        )


class StartControlWrapper(BaseWrapper):
    def __init__(self, env):
        self.env = env
        super.__init__(env)
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self) -> torch.Tensor:
        time.sleep(1)
        print("\nSet the robot in the starting position.")
        time.sleep(1)
        start_verifier = input("Press enter to start the environment.")
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)


class StackObservationsWrapper(gym.Wrapper):
    def __init__(self, env: Env, stack_size: int = 5):
        super().__init__(env)
        self.stack_size = stack_size
        self.observation_space = gym.spaces.Box(
            low=env.observation_space.low.repeat(stack_size, axis=-1),
            high=env.observation_space.high.repeat(stack_size, axis=-1),
            dtype=env.observation_space.dtype,
        )
        self.observation_buffer = np.zeros(
            (self.observation_space.shape), dtype=self.observation_space.dtype
        )

    def reset(self):
        obs = self.env.reset()
        self.observation_buffer[..., : self.env.observation_space.shape[-1]] = obs
        return self.observation_buffer

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.observation_buffer[
            ..., : -self.env.observation_space.shape[-1]
        ] = self.observation_buffer[..., self.env.observation_space.shape[-1] :]
        self.observation_buffer[..., -self.env.observation_space.shape[-1] :] = obs
        return self.observation_buffer, reward, done, info


class ActionFilterWrapper(gym.ActionWrapper):
    def __init__(self, env: Env, current_action_influence: float = 0.2):
        super().__init__(env)
        self.current_action_influence = current_action_influence

    def action(self, action: np.array) -> np.array:
        filtered_action = (
            self.current_action_influence * action
            + (1 - self.current_action_influence) * self.prev_action
        )
        self.prev_action = filtered_action
        return filtered_action

    def reset(self):
        self.prev_action = self.action_space.sample()
        return self.env.reset()


class FrameSkipWrapper(gym.Wrapper):
    def __init__(self, env, frame_skip):
        super().__init__(env)
        self.frame_skip = frame_skip
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        return self.env.reset()

    def step(self, action):
        total_reward = 0
        done = False
        for i in range(self.frame_skip):
            observation, reward, done, info = self.env.step(action)
            total_reward += reward
            self.current_step += 1
            if done:
                break
        return observation, total_reward, done, info
