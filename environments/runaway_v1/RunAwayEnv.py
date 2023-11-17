import time
from typing import Tuple

import gym
import numpy as np

from environments.base.base_env import BaseEnv


class RunAwayEnv_v1(BaseEnv):
    """
    A reinforcement learning environment for training agents to get away from a wall.
    This is the v1 version of the environment which is a bit more difficult where you have to control the wheel motors independently.

    The goal of the agent is to increase the distance measured by an ultrasonic sensor and get away from the wall as fast as possible.
    The environment provides a state consisting of 4 sensor readings (left, right, pitch, roll) and the distance to the wall.
    The agent can take a continuous action in the range [-1, 1] to control the movement of the robot.
    The environment returns a reward based on the change in distance to the wall and terminates the episode if the robot gets too close to the wall or the maximum number of steps is reached.

    Args:
        max_episode_steps (int): The maximum number of steps per episode. Defaults to 10.
        max_distance (float): The maximum distance to the wall. Defaults to 1000.0.
        min_distance (float): The minimum distance to the wall. Defaults to 40.
        sleep_time (float): The time to wait between sending actions and receiving the next state. Defaults to 0.2.

    Attributes:
        action_space (gym.spaces.Box): The continuous action space in the range [-1, 1].
        observation_space (gym.spaces.Box): The state space consisting of 4 sensor readings and the distance to the wall.

    Methods:
        sample_random_action(): Samples a random action from the action space.
        normalize_state(state: np.ndarray) -> np.ndarray: Normalizes and clips the state to be compatible with the agent.
        reset() -> np.ndarray: Resets the environment and returns the initial state.
        reward(state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> Tuple[float, bool]: Calculates the reward based on the change in distance to the wall.
        step(action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]: Performs the given action and returns the next state, reward, and done status.
    """

    def __init__(
        self,
        max_episode_steps: int = 10,
        max_distance: float = 1000.0,
        min_distance: float = 40,
        sleep_time: float = 0.2,
    ):
        action_dim = 2 # to control the wheel motors independently
        state_dim = 5  # 4 sensors (left,right,pitch,roll) + 1 distance to the wall
        self.sleep_time = sleep_time
        self.normalize_factor = 1000.0
        self.max_distance = max_distance
        self.min_distance = min_distance / self.normalize_factor

        self.max_episode_steps = max_episode_steps

        self.action_space = gym.spaces.Box(
            low=-np.ones(action_dim), high=np.ones(action_dim), shape=(action_dim,)
        )

        self.observation_space = gym.spaces.Box(
            low=np.zeros(state_dim),
            high=np.ones(state_dim) * self.max_distance,
            shape=(state_dim,),
        )

        super().__init__(action_dim=action_dim, state_dim=state_dim)

    def sample_random_action(self) -> np.ndarray:
        """
        Sample a random action from the action space.

        Returns:
            np.ndarray: A random action from the action space.
        """
        action = np.random.uniform(
            self.action_space.minimum, self.action_space.maximum, size=self.action_dim
        )
        return action

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """
        Normalize and clip the state to be compatible with the agent.

        Args:
            state (np.ndarray): The state to be normalized and clipped.

        Returns:
            np.ndarray: The normalized and clipped state.
        """
        state = np.clip(state, 0, self.max_distance)
        state = state / self.normalize_factor
        return state

    def reset(self) -> np.ndarray:
        """
        Reset the environment and return the initial state.

        Returns:
            np.ndarray: The initial state of the environment.
        """
        # TODO solve this fake action sending before to receive first state
        self.episode_step_iter = 0
        action = np.zeros(self.action_dim)
        self.send_to_hub(action)
        time.sleep(self.sleep_time)
        self.observation = self.normalize_state(self.read_from_hub())

        return self.observation.squeeze()

    def reward(
        self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray
    ) -> Tuple[float, bool]:
        """Reward function of RunAwayEnv.

        Goal: Increase distance measured by ultrasonic sensor aka.
        get away from the wall as fast as possible.

        """
        # TODO: maybe add reward for low motor usage (energy efficiency) so that the robot relaxes when max distance is reached
        done = False

        if (
            next_state[:, -1] <= self.min_distance
        ):  # too close to the wall break episode
            reward = -10
            done = True
        elif next_state[:, -1] < state[:, -1]:
            reward = -1.0
        elif next_state[:, -1] > state[:, -1]:
            reward = 1.0
        else:
            reward = 0.0

        return reward, done

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Perform the given action and return the next state, reward, and done status.

        Args:
            action (np.ndarray): The action to perform.

        Returns:
            Tuple[np.ndarray, float, bool, dict]: A tuple containing the next state, the reward
            received for performing the action, a boolean indicating whether the episode is done,
            and an empty dictionary.
        """

        # Send action to hub to receive next state
        self.send_to_hub(action)
        time.sleep(
            self.sleep_time
        )  # we need to wait some time for sensors to read and to
        # receive the next state
        next_observation = self.normalize_state(self.read_from_hub())

        # calc reward and done
        reward, done = self.reward(
            state=self.observation, action=action, next_state=next_observation
        )

        print("Reward", reward)
        # set next state as current state
        self.observation = next_observation

        # increment episode step counter
        self.episode_step_iter += 1
        if self.episode_step_iter >= self.max_episode_steps:
            done = True

        return self.observation.squeeze(), reward, done, {}
