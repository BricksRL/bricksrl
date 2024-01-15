import time
from typing import Tuple

import gym
import numpy as np

from environments.base.base_env import BaseEnv


class WalkerEnv_v2(BaseEnv):
    """
    A reinforcement learning environment for the robodog to learn to walk.
    This version of the Walker environment uses the acceleration and the distance as a reward. The distance to an object should be minimized.
    
    The reward is calculated as follows:
    reward_acc = -acc_x * dt
    reward_dist = 1 if new_dist < old_dist else -1
    reward = reward_acc + reward_dist

    
    
    Args:
        max_episode_steps (int): The maximum number of steps per episode. Defaults to 10.
        max_acc (float): The maximum acceleration of the robot. Defaults to 3000.0.
        max_distance (float): The maximum distance to the wall. Defaults to 2000.0.
        use_acceleration_reward (bool): Whether to use the acceleration reward. Defaults to True.
        reward_normalization_factor (float): The factor used to normalize the reward. Defaults to 1000.0.
        reward_clip_acc (bool): Whether to clip the acceleration reward to [-1, 1]. Defaults to False.
        sleep_time (float): The time to wait between sending actions and receiving the next state. Defaults to 0.0.
        verbose (bool): Whether to print additional information. Defaults to False.

    Attributes:
        action_space (gym.spaces.Box): The continuous action space in the range [-1, 1].
        observation_space (gym.spaces.Box): The state space consisting of 4 sensor readings and the distance to the wall.

    Methods:
        sample_random_action(): Samples a random action from the action space.
        normalize_state(state: np.ndarray) -> np.ndarray: Normalizes and clips the state to be compatible with the agent.
        reset() -> np.ndarray: Resets the environment and returns the initial state.
        reward(state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> Tuple[float, bool]: Calculates the reward based on the change in distance to the wall.
        step(action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]: Performs the given action and returns the next state, reward, done status, and truncation status.
    """

    def __init__(
        self,
        max_episode_steps: int = 50,
        max_acc: float = 3000.0,
        max_distance: float = 2000.0,
        reward_normalization_factor: float = 1000.0,
        reward_clip_acc: bool = False,
        use_acceleration_reward: bool = True,
        sleep_time: float = 0.0,
        verbose: bool = False,
    ):
        action_dim = 4 # (lf_value, lb_value, rf_value, rb_value)
        # angles are in range [-180, 179] 

        state_dim = 8  # (lf_angle, rf_angle, lb_angle, rb_angle, pitch, roll, acc_x, distance)
        self.sleep_time = sleep_time
        self.normalize_factor = reward_normalization_factor
        self.max_acc = max_acc
        self.max_distance = max_distance
        self.reward_clip = reward_clip_acc
        self.use_acceleration_reward = use_acceleration_reward


        self.max_episode_steps = max_episode_steps

        self.action_space = gym.spaces.Box(
            low=-np.ones(action_dim), high=np.ones(action_dim), shape=(action_dim,)
        )
        motor_range = (-179, 179)
        pitch_roll_range = (-50, 50)
        max_acc_range = (-self.max_acc, self.max_acc)
        dist_range = (0, self.max_distance)
        self.observation_space = gym.spaces.Box(
            low=np.array([motor_range[0], motor_range[0], motor_range[0], motor_range[0], pitch_roll_range[0], pitch_roll_range[0], max_acc_range[0], dist_range[0]]),
            high=np.array([motor_range[1], motor_range[1], motor_range[1], motor_range[1], pitch_roll_range[1], pitch_roll_range[1], max_acc_range[1], dist_range[1]]),
        )

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
        state = (state - self.observation_space.low) / (self.observation_space.high - self.observation_space.low)
        return state

    def reset(self) -> np.ndarray:
        """
        Reset the environment and return the initial state.

        Returns:
            np.ndarray: The initial state of the environment.
        """
        # TODO solve this fake action sending before to receive first state
        self.episode_step_iter = 0
        action = np.zeros(self.action_dim) + 1 # to bring robot in starting position!
        self.send_to_hub(action)
        time.sleep(self.sleep_time)
        self.observation = self.read_from_hub()

        if self.verbose:
            print("Raw state received: ", self.observation)
        self.dt = time.time()
        return self.normalize_state(self.observation.squeeze())

    def reward(
        self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray, delta_t: float
    ) -> Tuple[float, bool]:
        """Reward function of walker.

        Goal: Increase forward velocity, estimated from acceleration.

        Args:
            state (np.ndarray): The current state.
            action (np.ndarray): The action taken.
            next_state (np.ndarray): The next state.
            delta_t (float): The time step duration.

        Returns:
            Tuple[float, bool]: The reward received and a boolean indicating whether the episode is done.
        """

        done = False
        # pitch and roll need to stay in range [-75, 75] outside done = True
        pitch, roll = next_state[:, -4], next_state[:, -3]
        if np.abs(pitch) > 100 or np.abs(roll) > 100:
            done = True
            reward = 0
            return reward, done

        # distance reward
        prev_dist = state[:, -1]
        new_dist = next_state[:, -1]
        reward_dist = 1 if new_dist < prev_dist else -1

        if self.reward_clip:
            reward = np.where(next_state[:, -2] < 0, 1, -1)
        else:
            # Change in velocity (Δv = a * dt)
            reward = - next_state[:, -2]  * delta_t
            reward = reward / self.normalize_factor

        if self.use_acceleration_reward:
            reward = reward + reward_dist
        else:
            reward = reward_dist

        return reward.item(), done

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Perform the given action and return the next state, reward, done status, and truncation status.

        Args:
            action (np.ndarray): The action to perform.

        Returns:
            Tuple[np.ndarray, float, bool, bool, dict]: A tuple containing the next state, the reward
            received for performing the action, a boolean indicating whether the episode is done,
            a boolean indicating whether the episode is truncated, and an empty dictionary.
        """
        truncated = False
        # Send action to hub to receive next state
        self.send_to_hub(action)
        time.sleep(
            self.sleep_time
        )  # we need to wait some time for sensors to read and to
        
        # receive the next state
        next_observation = self.read_from_hub()
        current_time = time.time()
        delta_t = current_time - self.dt
        # calc reward and done
        reward, done = self.reward(
            state=self.observation, action=action, next_state=next_observation, delta_t=delta_t
        )
        if self.verbose:
            print("State", self.observation)
            print("Action", action)
            print("Next state", next_observation)
            print("Reward", reward)
            print("Step time", delta_t)
        # set next state as current state
        self.observation = self.normalize_state(next_observation)

        # increment episode step counter
        self.episode_step_iter += 1
        if self.episode_step_iter >= self.max_episode_steps:
            truncated = True
        self.dt = current_time
        return self.observation.squeeze(), reward, done, truncated, {"step_time": delta_t}
