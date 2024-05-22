from typing import Tuple

import numpy as np
import torch

from environments.base.base_env import BaseSimEnv
from tensordict import TensorDict, TensorDictBase
from torchrl.data.tensor_specs import BoundedTensorSpec, CompositeSpec


class RoboArmSimEnv_v0(BaseSimEnv):
    """ """

    action_dim = 4  # (Grab_motor_action, high_motor_action, low_motor_action, rotation_motor_action)

    state_dim = 4  # (GM, HM, LM, RM)

    motor_ranges = {
        "GM": (-148, -44),
        "HM": (-150, 10),
        "LM": (10, 70),
        "RM": (-180, 179),
    }

    observation_key = "vec_observation"
    goal_observation_key = "goal_state"
    original_observation_key = "original_state"
    original_goal_observation_key = "original_goal_state"

    def __init__(
        self,
        max_episode_steps: int = 50,
        noise: float = 0.1,
        verbose: bool = False,
        reward_signal: str = "dense",
    ):
        self.noise = noise

        assert reward_signal in [
            "dense",
            "sparse",
        ], "Reward signal must be dense or sparse."
        self.reward_signal = reward_signal
        self.max_episode_steps = max_episode_steps
        self._batch_size = torch.Size([1])
        self.current_position = None

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
                self.motor_ranges["GM"],
                self.motor_ranges["HM"],
                self.motor_ranges["LM"],
                self.motor_ranges["RM"],
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
            action_dim=self.action_dim, state_dim=self.state_dim, verbose=verbose
        )

    def normalize_state(self, state: np.ndarray, key: str) -> torch.Tensor:
        """
        Normalize the state to be processed by the agent.

        Args:
            state (np.ndarray): The state to be normalized.

        Returns:
            torch.Tensor: The normalized state.
        """
        state = (torch.from_numpy(state) - self.observation_spec[key].space.low) / (
            self.observation_spec[key].space.high - self.observation_spec[key].space.low
        )
        return state

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        """
        Reset the environment and return the initial state.

        Returns:
            TensorDictBase: The initial state of the environment.
        """
        # TODO solve this fake action sending before to receive first state
        self.episode_step_iter = 0

        observation = self.observation_spec[self.observation_key].rand().numpy()
        self.current_position = observation
        norm_obs = self.normalize_state(observation, self.observation_key)
        # sample random goal state
        goal_state = self.observation_spec[self.goal_observation_key].rand().numpy()
        norm_goal = self.normalize_state(goal_state, self.goal_observation_key)

        return TensorDict(
            {
                self.observation_key: norm_obs.float(),
                self.goal_observation_key: norm_goal.float(),
                self.original_observation_key: torch.from_numpy(observation).float(),
                self.original_goal_observation_key: torch.from_numpy(
                    goal_state
                ).float(),
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
        goal_state: np.array,
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
                goal_state, achieved_state
            )
            error = np.sum(np.abs(angle_deltas))
            reward = -error / 100
            if error < np.mean(self.goal_thresholds):
                done = True
        elif self.reward_signal == "sparse":
            angle_deltas = self.shortest_angular_distance_vectorized(
                goal_state, achieved_state
            )
            error = np.sum(np.abs(angle_deltas))
            if np.all(error <= self.goal_thresholds):
                reward = 1
                done = True

            else:
                reward = 0
        else:
            raise ValueError("Reward signal must be dense or sparse.")

        return reward, done

    @staticmethod
    def transform_range(value, old_min, old_max, new_min, new_max):
        """
        Transform a value from one range to another.

        Parameters:
        value (float): The value to transform.
        old_min (float): The minimum value of the old range.
        old_max (float): The maximum value of the old range.
        new_min (float): The minimum value of the new range.
        new_max (float): The maximum value of the new range.

        Returns:
        float: The transformed value.
        """
        # Compute the scale factor between the old and new ranges
        scale = (new_max - new_min) / (old_max - old_min)
        # Apply the transformation
        return new_min + (value - old_min) * scale

    @staticmethod
    def normalize_angle(angle, low_angle=-180, high_angle=179, original_one_round=360):
        # Normalize angle to be within -179 to 179 degrees
        while angle <= low_angle:
            angle += original_one_round
        while angle > high_angle:
            angle -= original_one_round
        return angle

    def apply_action(self, action: np.ndarray) -> np.ndarray:

        noise = np.random.normal(0, self.noise, size=4)
        action += noise

        rotation_action, low_action, high_action, grab_action = action
        # transform action range for motors
        grab_action = self.transform_range(grab_action, -1, 1, -25, 25)
        high_action = self.transform_range(high_action, -1, 1, -60, 60)
        low_action = self.transform_range(low_action, -1, 1, -30, 30)
        rotation_action = self.transform_range(rotation_action, -1, 1, -180, 180)

        (
            current_grab_angle,
            current_high_angle,
            current_low_angle,
            current_rotation_angle,
        ) = self.current_position.squeeze()

        if not (
            current_grab_angle + grab_action > max(self.motor_ranges["GM"])
        ) and not (current_grab_angle + grab_action < min(self.motor_ranges["GM"])):
            current_grab_angle += grab_action

        if not (
            current_high_angle + high_action > max(self.motor_ranges["HM"])
        ) and not (current_high_angle + high_action < min(self.motor_ranges["HM"])):
            current_high_angle += high_action

        if not (current_low_angle + low_action > max(self.motor_ranges["LM"])) and not (
            current_low_angle + low_action < min(self.motor_ranges["LM"])
        ):
            current_low_angle += low_action

        current_rotation_angle += rotation_action
        current_rotation_angle = self.normalize_angle(current_rotation_angle)

        self.current_position = np.array(
            [
                [
                    current_grab_angle,
                    current_high_angle,
                    current_low_angle,
                    current_rotation_angle,
                ]
            ]
        )
        return self.current_position

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """ """
        # Send action to hub to receive next state
        action = tensordict.get("action").cpu().numpy().squeeze()

        next_observation = self.apply_action(action)
        goal_state = tensordict.get(self.original_goal_observation_key).cpu().numpy()

        # calc reward and done
        reward, done = self.reward(
            achieved_state=next_observation,
            goal_state=goal_state,
        )

        next_tensordict = TensorDict(
            {
                self.observation_key: self.normalize_state(
                    next_observation, self.observation_key
                ).float(),
                self.original_observation_key: torch.from_numpy(
                    next_observation
                ).float(),
                self.goal_observation_key: tensordict.get(
                    self.goal_observation_key
                ).float(),
                self.original_goal_observation_key: tensordict.get(
                    self.original_goal_observation_key
                ).float(),
                "reward": torch.tensor([reward]).float(),
                "done": torch.tensor([done]).bool(),
            },
            batch_size=[1],
        )

        # increment episode step counter
        self.episode_step_iter += 1
        if self.episode_step_iter >= self.max_episode_steps:
            next_tensordict.set("done", torch.tensor([True]))
        return next_tensordict
