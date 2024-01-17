import time
from typing import Tuple

import gym
import numpy as np

from environments.base.base_env import BaseEnv


class SpinningEnv_v1(BaseEnv):
    """
    SpinningEnv_v1 is a custom gym environment for a spinning robot.
    The robot has to learn to spin in a circle around its own axis given a random goal direction (left or right, 0 or 1).

    Args:
        max_episode_steps (int): The maximum number of steps per episode. Defaults to 50.
        sleep_time (float): The time to wait between sending actions and receiving the next state. Defaults to 0.2.
        verbose (bool): Whether to print verbose information during the environment's execution. Defaults to False.

    Attributes:
        action_space (gym.spaces.Box): The continuous action space in the range [-1, 1].
        observation_space (gym.spaces.Box): The state space consisting of 5 sensor readings (left, right, pitch, roll, rotation_velocity) and 1 direction (left or right).

    Methods:
        sample_random_action() -> np.ndarray: Samples a random action from the action space.
        normalize_state(state: np.ndarray) -> np.ndarray: Normalizes and clips the state to be compatible with the agent.
        reset() -> np.ndarray: Resets the environment and returns the initial state.
    """

    def __init__(
        self,
        max_episode_steps: int = 50,
        sleep_time: float = 0.2,
        verbose: bool = False,
    ):
        action_dim = 2  # to control the wheel motors independently
        state_dim = 5  # 5 sensors (left,right,pitch,roll, rotation_velocity) + 1 direction (left or right)

        motor_angles = (0, 360)
        pitch_angles = (-90, 90)
        roll_angles = (-90, 90)
        rotation_velocity = (-100, 100) # adapt to real values

        self.sleep_time = sleep_time

        self.max_episode_steps = max_episode_steps

        self.action_space = gym.spaces.Box(
            low=-np.ones(action_dim), high=np.ones(action_dim), shape=(action_dim,)
        )

        self.observation_space = gym.spaces.Box(
            low=np.array(
                [
                    motor_angles[0],
                    motor_angles[0],
                    pitch_angles[0],
                    roll_angles[0],
                    rotation_velocity[0],
                    0,
                ]
            ),
            high=np.array(
                [
                    motor_angles[1],
                    motor_angles[1],
                    pitch_angles[1],
                    pitch_angles[1],
                    rotation_velocity[1],
                    1,
                ]
            ),
        )
        self.verbose = verbose
        super().__init__(action_dim=action_dim, state_dim=state_dim, verbose=verbose)

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
        state = np.clip(state, self.observation_space.low, self.observation_space.high)
        state = (state - self.observation_space.low) / (
            self.observation_space.high - self.observation_space.low
        )
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

        state = self.read_from_hub()
        self.direction = np.random.randint(0, 2)  # (0,1) left or right
        self.observation = self.normalize_state(np.concatenate((state, np.array([[self.direction]])), axis=1))

        return self.observation.squeeze()

    def reward(self, next_state: np.ndarray) -> Tuple[float, bool]:
        """Reward function of Spinning environment.
        If the self.direction is 0, the robot is spinning left, otherwise right.
        We want to maximise in those cases the angular velocity (last element of the state vector).
        If the robot is spinning in the wrong direction, we want to minimize the angular velocity.
        """
        # TODO: maybe add reward for low motor usage (energy efficiency) so that the robot relaxes when max distance is reached
        done = False
        velocity = next_state[:, -1]

        if self.direction == 0:
            reward = velocity
        else:
            reward = -velocity

        return reward.item(), done

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
        truncated = False
        # Send action to hub to receive next state
        self.send_to_hub(action)
        time.sleep(
            self.sleep_time
        )  # we need to wait some time for sensors to read and to
        # receive the next state
        next_observation = self.read_from_hub()

        # calc reward and done
        reward, done = self.reward(next_state=next_observation)
        if self.verbose:
            print("Action", action)
            print("Old distance", self.observation[:, -1])
            print("New distance", next_observation[:, -1])
            print("Reward", reward)
        # set next state as current state
        self.observation = self.normalize_state(np.concatenate((next_observation, np.array([[self.direction]])), axis=1))

        # increment episode step counter
        self.episode_step_iter += 1
        if self.episode_step_iter >= self.max_episode_steps:
            truncated = True

        return self.observation.squeeze(), reward, done, truncated, {}
