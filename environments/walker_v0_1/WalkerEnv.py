import time
from typing import Tuple

import gym
import numpy as np

from environments.base.base_env import BaseEnv


class WalkerEnv_v01(BaseEnv):
    """
    A reinforcement learning environment for the robodog to learn to walk.

    Specific to the walker_v0 environment is, that the reward function is hard coded to learn a gait routine.
    In contrast to the walker_v1 environment, the reward function is not based on the acceleration of the robot.

    Args:
        max_episode_steps (int): The maximum number of steps per episode. Defaults to 10.
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
        sleep_time: float = 0.0,
        verbose: bool = False,
    ):
        action_dim = 8  # (lf_value, lb_value, rf_value, rb_value, lf_speed, lb_speed, rf_speed, rb_speed)
        # angles are in range [-180, 179]
        state_dim = 7  # (lf_angle, rf_angle, lb_angle, rb_angle, pitch, roll, acc_x)
        self.sleep_time = sleep_time
        self.max_acc = 3000

        self.max_episode_steps = max_episode_steps

        self.action_space = gym.spaces.Box(
            low=-np.ones(action_dim), high=np.ones(action_dim), shape=(action_dim,)
        )
        motor_range = (-179, 179)
        pitch_roll_range = (-50, 50)
        max_acc_range = (-self.max_acc, self.max_acc)
        self.observation_space = gym.spaces.Box(
            low=np.array(
                [
                    motor_range[0],
                    motor_range[0],
                    motor_range[0],
                    motor_range[0],
                    pitch_roll_range[0],
                    pitch_roll_range[0],
                    max_acc_range[0],
                ]
            ),
            high=np.array(
                [
                    motor_range[1],
                    motor_range[1],
                    motor_range[1],
                    motor_range[1],
                    pitch_roll_range[1],
                    pitch_roll_range[1],
                    max_acc_range[1],
                ]
            ),
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
        action = np.zeros(self.action_dim) + 1  # to bring robot in starting position!
        self.send_to_hub(action)
        time.sleep(self.sleep_time)
        self.observation = self.read_from_hub()

        if self.verbose:
            print("Raw state received: ", self.observation)
        self.dt = time.time()
        return self.normalize_state(self.observation.squeeze())

    def reward(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        delta_t: float,
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
        pitch, roll = next_state[:, -3], next_state[:, -2]
        if np.abs(pitch) > 100 or np.abs(roll) > 100:
            done = True
            reward = 0
            return reward, done

        (
            lf_angle,
            rf_angle,
            lb_angle,
            rb_angle,
            pitch,
            roll,
            acc_x,
        ) = next_state.squeeze()

        # we want actions to be negative and high
        # action is in range [-1, 1] over 4 dims -> sum is in range [-4, 4] -> divide by 4 to get in range [-1, 1]
        action_reward = -np.sum(action) / 4 / 10
        # Take this off we dont want them to be similar otherwise we cant adapt for noise in the system
        # actions should ideally be similar something like [-0.75, -0.75, -0.75, -0.75]
        # action_std_reward = -np.std(action)

        # we want lf_angle and rb_angle to be synchronized and rf_angle and lb_angle to be synchronized
        # divide by 180 to get in range [-1, 0]
        lf_rb_diff_reward = -angular_difference(lf_angle, rb_angle) / 180
        rf_lb_diff_reward = -angular_difference(rf_angle, lb_angle) / 180

        # we want lf_rb and rf_lb to be 180° apart
        # divide by 180 to get in range [-1, 0]
        lf_rf_180_reward = -(180 - angular_difference(lf_angle, rf_angle)) / 180
        lb_rb_180_reward = -(180 - angular_difference(lb_angle, rb_angle)) / 180

        if self.verbose:
            # TODO: maybe we want add those values as an info dict to the env step return
            print("action_reward", action_reward)
            # print("action_std_reward", action_std_reward)
            print("lf_rb_diff_reward", lf_rb_diff_reward)
            print("rf_lb_diff_reward", rf_lb_diff_reward)
            print("lf_rf_180_reward", lf_rf_180_reward)

        reward = (
            action_reward
            # + action_std_reward
            + lf_rb_diff_reward
            + rf_lb_diff_reward
            + lf_rf_180_reward
            + lb_rb_180_reward
        )

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
            state=self.observation,
            action=action,
            next_state=next_observation,
            delta_t=delta_t,
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
        return (
            self.observation.squeeze(),
            reward,
            done,
            truncated,
            {"step_time": delta_t},
        )


def angular_difference(angle1, angle2):
    # Calculate the difference in angles, wrapped between -180 and 180
    difference = (angle2 - angle1 + 180) % 360 - 180
    return abs(difference)  # Return the absolute value of the difference
