import time
from typing import Tuple

import numpy as np
import torch

from bricksrl.environments.base.base_env import BaseEnv
from numpy import linalg
from tensordict import TensorDict, TensorDictBase
from torchrl.data.tensor_specs import BoundedTensorSpec, CompositeSpec


class RoboArmEnv_v0(BaseEnv):
    """ """

    action_dim = 4  # (Grab_motor_action, high_motor_action, low_motor_action, rotation_motor_action)

    state_dim = 4  # (GM, HM, LM, RM)

    observation_ranges = {
        "GM": (-148, -44),
        "HM": (-150, 10),
        "LM": (10, 70),
        "RM": (-180, 179),
    }

    observation_key = "observation"
    goal_observation_key = "goal_observation"

    def __init__(
        self,
        max_episode_steps: int = 50,
        sleep_time: float = 0.0,
        verbose: bool = False,
        pretrain: bool = False,
        reward_signal: str = "dense",
    ):
        self.sleep_time = sleep_time

        assert reward_signal in [
            "dense",
            "sparse",
        ], "Reward signal must be dense or sparse."
        self.reward_signal = reward_signal
        self.max_episode_steps = max_episode_steps
        self._batch_size = torch.Size([1])

        # Define action spec
        self.action_spec = BoundedTensorSpec(
            low=-1,
            high=1,
            shape=(1, self.action_dim),
        )

        self.goal_thresholds = np.array(
            [50]
        )  # everythin below 20 is very good. 50 is good!
        # Observation 4 motors (GM, HM, LM, RM) + goal positions (GGM, GHM, GLM, GRM)
        # Define observation spec
        bounds = torch.tensor(
            [
                self.observation_ranges["GM"],
                self.observation_ranges["HM"],
                self.observation_ranges["LM"],
                self.observation_ranges["RM"],
            ]
        )

        low_bounds = bounds[:, 0].unsqueeze(0)
        high_bounds = bounds[:, 1].unsqueeze(0)

        observation_spec = BoundedTensorSpec(
            low=low_bounds,
            high=high_bounds,
        )

        self.observation_spec = CompositeSpec(shape=(1,))
        self.observation_spec.set(self.observation_key, observation_spec)
        self.observation_spec.set(self.goal_observation_key, observation_spec)
        super().__init__(
            action_dim=self.action_dim,
            state_dim=self.state_dim,
            verbose=verbose,
            use_hub=1 - pretrain,
        )

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        """
        Reset the environment and return the initial state.

        Returns:
            TensorDictBase: The initial state of the environment.
        """
        # TODO solve this fake action sending before to receive first state
        self.episode_step_iter = 0
        if tensordict is not None:
            action = tensordict.get("action").cpu().numpy().squeeze()
        else:
            action = np.zeros(self.action_dim)
        self.send_to_hub(action)
        time.sleep(self.sleep_time)
        observation = self.read_from_hub()
        # sample random goal state
        self.goal_observation = (
            self.observation_spec[self.goal_observation_key].rand().numpy()
        )

        return TensorDict(
            {
                self.observation_key: torch.tensor(observation, dtype=torch.float32),
                self.goal_observation_key: torch.tensor(
                    self.goal_observation, dtype=torch.float32
                ),
                "error": torch.tensor([0]).float(),
            },
            batch_size=[1],
        )

    @staticmethod
    def shortest_angular_distance_vectorized(
        theta_goal: np.array, theta_current: np.array
    ) -> float:
        """
        Calculate the shortest angular distance between two arrays of angles.

        Parameters:
        - theta_goal: Array of goal angles in degrees.
        - theta_current: Array of current angles in degrees.

        Returns:
        - Array of the shortest angular distances in degrees.
        """

        # Convert angles from degrees to radians
        theta_goal_rad = np.radians(theta_goal)
        theta_current_rad = np.radians(theta_current)

        # Calculate difference in radians using np.arctan2 for vectorized operation
        delta_theta_rad = np.arctan2(
            np.sin(theta_goal_rad - theta_current_rad),
            np.cos(theta_goal_rad - theta_current_rad),
        )

        # Convert result back to degrees
        delta_theta_deg = np.degrees(delta_theta_rad)

        return delta_theta_deg

    def reward(
        self,
        achieved_state: np.array,
    ) -> Tuple[float, bool]:
        """Reward function of roboarm.

        Args:
            achieved_state (np.ndarray): The achieved state.
            goal_state (np.ndarray): The goal state.

        Returns:
            Tuple[float, bool]: The reward received and a boolean indicating whether the episode is done.
        """

        done = False
        if self.reward_signal == "dense":
            angle_deltas = self.shortest_angular_distance_vectorized(
                self.goal_observation, achieved_state
            )
            error = np.sum(np.abs(angle_deltas))
            reward = -error / 100
            if error < np.mean(self.goal_thresholds):
                done = True
        elif self.reward_signal == "sparse":
            angle_deltas = self.shortest_angular_distance_vectorized(
                self.goal_observation, achieved_state
            )
            error = np.sum(np.abs(angle_deltas))
            if np.all(error <= self.goal_thresholds):
                reward = 1
                done = True

            else:
                reward = 0
        else:
            raise ValueError("Reward signal must be dense or sparse.")

        return reward, done, error

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """ """
        # Send action to hub to receive next state
        self.send_to_hub(tensordict.get("action").cpu().numpy().squeeze())
        time.sleep(
            self.sleep_time
        )  # we need to wait some time for sensors to read and to

        # receive the next state
        next_observation = self.read_from_hub()

        # calc reward and done
        reward, done, error = self.reward(
            achieved_state=next_observation,
        )

        next_tensordict = TensorDict(
            {
                self.observation_key: torch.tensor(
                    next_observation, dtype=torch.float32
                ),
                self.goal_observation_key: torch.tensor(
                    self.goal_observation, dtype=torch.float32
                ),
                "reward": torch.tensor([reward]).float(),
                "done": torch.tensor([done]).bool(),
                "error": torch.tensor([error]).float(),
            },
            batch_size=[1],
        )

        # increment episode step counter
        self.episode_step_iter += 1
        if self.episode_step_iter >= self.max_episode_steps:
            next_tensordict.set("done", torch.tensor([True]))
        return next_tensordict
