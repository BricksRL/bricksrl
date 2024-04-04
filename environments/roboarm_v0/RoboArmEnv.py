import time
from typing import Tuple

import numpy as np
import torch

from environments.base.base_env import BaseEnv
from numpy import linalg
from tensordict import TensorDict, TensorDictBase
from torchrl.data.tensor_specs import BoundedTensorSpec, CompositeSpec


class RoboArmEnv_v0(BaseEnv):
    """ """

    action_dim = 4  # (Grab_motor_action, high_motor_action, low_motor_action, rotation_motor_action)

    state_dim = 4  # (GM, HM, LM, RM)

    motor_ranges = {
        "GM": (-148, -44),
        "HM": (-150, 10),
        "LM": (10, 70),
        "RM": (-900, 900),
    }

    observation_key = "vec_observation"
    goal_observation_key = "goal_state"
    original_observation_key = "original_state"
    original_goal_observation_key = "original_goal_state"

    def __init__(
        self,
        max_episode_steps: int = 50,
        sleep_time: float = 0.0,
        verbose: bool = False,
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

        self.action_spec = BoundedTensorSpec(
            low=-torch.ones((1, self.action_dim)),
            high=torch.ones((1, self.action_dim)),
            shape=(1, self.action_dim),
        )

        self.goal_thresholds = np.array([50])
        # Observation 4 motors (GM, HM, LM, RM) + goal positions (GGM, GHM, GLM, GRM)
        observation_spec = BoundedTensorSpec(
            low=torch.tensor(
                [
                    [
                        self.motor_ranges["GM"][0],
                        self.motor_ranges["HM"][0],
                        self.motor_ranges["LM"][0],
                        self.motor_ranges["RM"][0],
                    ]
                ]
            ),
            high=torch.tensor(
                [
                    [
                        self.motor_ranges["GM"][1],
                        self.motor_ranges["HM"][1],
                        self.motor_ranges["LM"][1],
                        self.motor_ranges["RM"][1],
                    ]
                ]
            ),
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
        if tensordict is not None:
            action = tensordict.get("action").cpu().numpy().squeeze()
        else:
            action = np.zeros(self.action_dim)
        self.send_to_hub(action)
        time.sleep(self.sleep_time)
        observation = self.read_from_hub()
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

    def reward(
        self,
        achieved_state: np.array,
        goal_state: np.array,
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
        if self.reward_signal == "dense":
            # error = np.sum(np.abs(achieved_state - goal_state))
            error = linalg.norm(achieved_state - goal_state)
            reward = -error / 1000
            if error < np.mean(self.goal_thresholds):
                done = True
        elif self.reward_signal == "sparse":
            errors = np.abs(achieved_state - goal_state)
            if np.all(errors <= self.goal_thresholds):
                reward = 1
                done = True

            else:
                reward = 0
        else:
            raise ValueError("Reward signal must be dense or sparse.")

        return reward, done

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """ """
        # Send action to hub to receive next state
        self.send_to_hub(tensordict.get("action").cpu().numpy().squeeze())
        time.sleep(
            self.sleep_time
        )  # we need to wait some time for sensors to read and to

        # receive the next state
        next_observation = self.read_from_hub()
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
