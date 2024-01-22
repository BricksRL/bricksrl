import time
from typing import Tuple

import gym
import numpy as np

from environments.base.base_env import BaseEnv


class RoboArmEnv_v0(BaseEnv):
    """ """

    def __init__(
        self,
        max_episode_steps: int = 50,
        sleep_time: float = 0.0,
        verbose: bool = False,
    ):
        action_dim = 2  # (Grab_motor_action, high_motor_action, low_motor_action, rotation_motor_action)
        # angles are in range [-180, 179]

        state_dim = 2  # (GM, HM, LM, RM, GGM, GHM, GLM, GRM)
        self.sleep_time = sleep_time


        self.max_episode_steps = max_episode_steps

        self.action_space = gym.spaces.Box(
            low=-np.ones(action_dim), high=np.ones(action_dim), shape=(action_dim,)
        )
        # grab motor range
        # (-230, -148) left side closed (-148, -44) -> starting -44

        # high motor range
        # (-60, 140) -> starting 123

        # low motor range
        # (-190, -10) -> starting -19

        # rotation motor range
        # (-360, 360) -> starting 45
        motor_ranges = {
            #"GM": (-230, -148),
            "HM": (-60, 140),
            #"LM": (-190, -10),
            "RM": (-900, 900),
        }
        self.motor_ranges = motor_ranges
        self.goal_thresholds = np.array([20, 20])
        # Observation 4 motors (GM, HM, LM, RM) + goal positions (GGM, GHM, GLM, GRM)
        self.observation_space = gym.spaces.Box(
            low=np.array(
                [
                    #self.motor_ranges["GM"][0],
                    self.motor_ranges["HM"][0],
                    #self.motor_ranges["LM"][0],
                    self.motor_ranges["RM"][0],
                    #self.motor_ranges["GM"][0],
                    self.motor_ranges["HM"][0],
                    #self.motor_ranges["LM"][0],
                    self.motor_ranges["RM"][0],
                ]
            ),
            high=np.array(
                [
                    #self.motor_ranges["GM"][1],
                    self.motor_ranges["HM"][1],
                    #self.motor_ranges["LM"][1],
                    self.motor_ranges["RM"][1],
                    #self.motor_ranges["GM"][1],
                    self.motor_ranges["HM"][1],
                    #self.motor_ranges["LM"][1],
                    self.motor_ranges["RM"][1],
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
        action = np.zeros(self.action_dim) # to bring robot in starting position!
        self.send_to_hub(action)
        time.sleep(self.sleep_time)
        self.observation = self.read_from_hub()

        # sample random goal state
        self.goal_state = np.random.uniform(
            low=[
                #self.motor_ranges["GM"][0],
                self.motor_ranges["HM"][0],
                #self.motor_ranges["LM"][0],
                self.motor_ranges["RM"][0],
            ],
            high=[
                #self.motor_ranges["GM"][1],
                self.motor_ranges["HM"][1],
                #self.motor_ranges["LM"][1],
                self.motor_ranges["RM"][1],
            ],
        )


        if self.verbose:
            print("Raw state received: ", self.observation)
            print("Goal state: ", self.goal_state)
        return self.normalize_state(np.concatenate([self.observation.squeeze(), self.goal_state]))

    def reward(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
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
        errors = np.abs(next_state - self.goal_state)
        if np.all(errors <= self.goal_thresholds):
            reward = 1
            done = True

        else:
            reward = 0

        return reward, done

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
        # calc reward and done
        reward, done = self.reward(
            state=self.observation,
            action=action,
            next_state=next_observation,
        )
        if self.verbose:
            print("State", self.observation)
            print("Action", action)
            print("Next state", next_observation)
            print("Reward", reward)
            print("Goal state", self.goal_state)
        # set next state as current state
        self.observation = next_observation

        # increment episode step counter
        self.episode_step_iter += 1
        if self.episode_step_iter >= self.max_episode_steps:
            truncated = True
        self.dt = current_time
        return (
            self.normalize_state(np.concatenate([self.observation.squeeze(), self.goal_state])),
            reward,
            done,
            truncated,
            {"desired_state": self.goal_state, "achieved_state": self.observation.squeeze()},
        )
