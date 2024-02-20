import time
from typing import Tuple

import numpy as np

import torch

from environments.base.base_env import BaseEnv
from tensordict import TensorDict, TensorDictBase
from torchrl.data.tensor_specs import BoundedTensorSpec, CompositeSpec


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

    """

    action_dim = 4  # (lf_value, lb_value, rf_value, rb_value)
    # angles are in range [-180, 179]

    state_dim = (
        8  # (lf_angle, rf_angle, lb_angle, rb_angle, pitch, roll, acc_x, distance)
    )
    motor_range = (-179, 179)
    pitch_roll_range = (-50, 50)
    observation_key = "observation_vector"
    original_vec_observation_key = "original_vec_observation"

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
        self.sleep_time = sleep_time
        self.normalize_factor = reward_normalization_factor
        self.max_acc = max_acc
        self.max_distance = max_distance
        self.reward_clip = reward_clip_acc
        self._batch_size = torch.Size([1])
        self.use_acceleration_reward = use_acceleration_reward

        self.max_episode_steps = max_episode_steps

        self.action_spec = BoundedTensorSpec(
            low=-torch.ones((1, self.action_dim)),
            high=torch.ones((1, self.action_dim)),
            shape=(1, self.action_dim),
        )

        max_acc_range = (-self.max_acc, self.max_acc)
        dist_range = (0, self.max_distance)

        observation_spec = BoundedTensorSpec(
            low=torch.tensor(
                [
                    [
                        self.motor_range[0],
                        self.motor_range[0],
                        self.motor_range[0],
                        self.motor_range[0],
                        self.pitch_roll_range[0],
                        self.pitch_roll_range[0],
                        max_acc_range[0],
                        dist_range[0],
                    ]
                ]
            ),
            high=torch.tensor(
                [
                    [
                        self.motor_range[1],
                        self.motor_range[1],
                        self.motor_range[1],
                        self.motor_range[1],
                        self.pitch_roll_range[1],
                        self.pitch_roll_range[1],
                        max_acc_range[1],
                        dist_range[1],
                    ]
                ]
            ),
        )
        self.observation_spec = CompositeSpec(shape=(1,))
        self.observation_spec.set(self.observation_key, observation_spec)
        super().__init__(
            action_dim=self.action_dim, state_dim=self.state_dim, verbose=verbose
        )

    def normalize_state(self, state: np.ndarray) -> torch.Tensor:
        """
        Normalize the state to be processed by the agent.

        Args:
            state (np.ndarray): The state to be normalized.

        Returns:
            torch.Tensor: The normalized state.
        """
        state = (
            torch.from_numpy(state)
            - self.observation_spec[self.observation_key].space.low
        ) / (
            self.observation_spec[self.observation_key].space.high
            - self.observation_spec[self.observation_key].space.low
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
            action = tensordict.get("action").numpy().squeeze()
        else:
            action = np.zeros(self.action_dim)
        self.send_to_hub(action)
        time.sleep(self.sleep_time)
        observation = self.read_from_hub()

        self.dt = time.time()
        norm_observation = self.normalize_state(observation)
        return TensorDict(
            {
                self.observation_key: norm_observation.float(),
                self.original_vec_observation_key: torch.from_numpy(
                    observation
                ).float(),
            },
            batch_size=[1],
        )

    def reward(
        self,
        state: np.ndarray,
        next_state: np.ndarray,
        delta_t: float,
    ) -> Tuple[float, bool]:
        """Reward function of walker.

        Goal: Increase forward velocity, estimated from acceleration.

        Args:
            state (np.ndarray): The current state.
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
            reward = -next_state[:, -2] * delta_t
            reward = reward / self.normalize_factor

        if self.use_acceleration_reward:
            reward = reward + reward_dist
        else:
            reward = reward_dist

        return reward.item(), done

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """ """
        # Send action to hub to receive next state
        action = tensordict.get("action").numpy().squeeze()
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
            state=tensordict.get(self.original_vec_observation_key).numpy(),
            next_state=next_observation,
            delta_t=delta_t,
        )

        next_tensordict = TensorDict(
            {
                self.observation_key: self.normalize_state(next_observation).float(),
                "original_vec_observation": torch.from_numpy(next_observation).float(),
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
