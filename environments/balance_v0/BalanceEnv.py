import time
from typing import Tuple

import gym
import numpy as np

from environments.base.base_env import BaseEnv
from tensordict import TensorDict, TensorDictBase
from torchrl.data.tensor_specs import BoundedTensorSpec, TensorSpec

class BalanceEnv_v0(BaseEnv):
    """
    BalanceEnv_v0 is a custom gym environment for the 2wheeler balancing robot.

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
    action_dim = 4  # to control the wheel motors and the motor speed
    state_dim = 4  # 4 sensors (left, right, roll, rotation_velocity)

    motor_angles = (0, 360)
    roll_angles = (-90, 90)
    rotation_velocity = (-250, 250) # adapt to real values

    def __init__(
        self,
        max_episode_steps: int = 50,
        sleep_time: float = 0.0,
        verbose: bool = False,
    ):

        self.sleep_time = sleep_time

        self.max_episode_steps = max_episode_steps

        self.action_spec = BoundedTensorSpec(
            low=-np.ones(self.action_dim), high=np.ones(self.action_dim), shape=(self.action_dim,)
        )

        self.observation_spec = BoundedTensorSpec(
            low=np.array(
                [
                    self.motor_angles[0],
                    self.motor_angles[0],
                    self.roll_angles[0],
                    self.rotation_velocity[0],
                ]
            ),
            high=np.array(
                [
                    self.motor_angles[1],
                    self.motor_angles[1],
                    self.roll_angles[1],
                    self.rotation_velocity[1],
                ]
            ),
        )
        self.verbose = verbose
        super().__init__(action_dim=self.action_dim, state_dim=self.state_dim, verbose=verbose)

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
            state (np.ndarray): The state to be normalized.

        Returns:
            np.ndarray: The normalized state.
        """
        state = np.clip(state, self.observation_space.low, self.observation_space.high)
        state = (state - self.observation_space.low) / (
            self.observation_space.high - self.observation_space.low
        )
        return state

    def reset(self) -> TensorDictBase:
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
        observation = self.normalize_state(self.read_from_hub())
        self.episode_return = 0 # reset episode return
        return TensorDict({"state": observation.squeeze(),
                           "episode_return": self.episode_return}, batch_size=[])

    def reward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Reward function of Spinning environment.
        If the self.direction is 0, the robot is spinning left, otherwise right.
        We want to maximise in those cases the angular velocity (last element of the state vector).
        If the robot is spinning in the wrong direction, we want to minimize the angular velocity.
        """
        # TODO: maybe add reward for low motor usage (energy efficiency) so that the robot relaxes when max distance is reached
        done = False
        velocity = next_state[:, -1]
        roll = next_state[:, -2]

        roll_reward = - np.abs(roll)

        # velocity_reward = - np.abs(velocity)/1000 # devide by 10 to put more focus on roll reward

        reward = roll_reward  #+ velocity_reward
        return reward.item(), done

    def step(self, tensordict: TensorDictBase) -> TensorDictBase:
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
        self.observation = self.normalize_state(next_observation)

        # increment episode step counter
        self.episode_step_iter += 1
        if self.episode_step_iter >= self.max_episode_steps:
            truncated = True

        return self.observation.squeeze(), reward, done, truncated, {}
