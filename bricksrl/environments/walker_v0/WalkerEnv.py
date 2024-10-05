import time
from typing import Tuple

import numpy as np

import torch

from bricksrl.environments.base.base_env import BaseEnv
from tensordict import TensorDict, TensorDictBase
from torchrl.data.tensor_specs import BoundedTensorSpec, CompositeSpec


class WalkerEnv_v0(BaseEnv):
    """
    A reinforcement learning environment for the robodog to learn to walk.

    Specific to the walker_v0 environment is, that the reward function is hard coded to learn a gait routine.
    In contrast to the walker_v1 environment, the reward function is not based on the acceleration of the robot.

    Args:
        max_episode_steps (int): The maximum number of steps per episode. Defaults to 10.
        sleep_time (float): The time to wait between sending actions and receiving the next state. Defaults to 0.0.
        verbose (bool): Whether to print additional information. Defaults to False.

    """

    action_dim = 4  # (lf_value, lb_value, rf_value, rb_value)
    # angles are in range [-179, 179]
    state_dim = 7  # (lf_angle, rf_angle, lb_angle, rb_angle, pitch, roll, acc_x)

    observation_ranges = {
        "lf_angle": [-179, 179],
        "rf_angle": [-179, 179],
        "lb_angle": [-179, 179],
        "rb_angle": [-179, 179],
        "pitch": [-50, 50],
        "roll": [-50, 50],
        "acc_x": [-3000, 3000],
    }

    observation_key = "observation"

    def __init__(
        self,
        max_episode_steps: int = 50,
        sleep_time: float = 0.0,
        verbose: bool = False,
        pretrain: bool = False,
    ):
        self.sleep_time = sleep_time
        self._batch_size = torch.Size([1])
        self.max_episode_steps = max_episode_steps

        # Define action spec
        self.action_spec = BoundedTensorSpec(
            low=-1,
            high=1,
            shape=(1, self.action_dim),
        )

        # Define observation spec
        bounds = torch.tensor(
            [
                self.observation_ranges["lf_angle"],
                self.observation_ranges["rf_angle"],
                self.observation_ranges["lb_angle"],
                self.observation_ranges["rb_angle"],
                self.observation_ranges["pitch"],
                self.observation_ranges["roll"],
                self.observation_ranges["acc_x"],
            ]
        )
        # Reshape bounds to (1, 7)
        low_bounds = bounds[:, 0].unsqueeze(0)
        high_bounds = bounds[:, 1].unsqueeze(0)

        observation_spec = BoundedTensorSpec(
            low=low_bounds,
            high=high_bounds,
            shape=(1, self.state_dim),
        )

        self.observation_spec = CompositeSpec(
            {self.observation_key: observation_spec}, shape=(1,)
        )
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

        return TensorDict(
            {
                self.observation_key: torch.tensor(observation, dtype=torch.float32),
            },
            batch_size=[1],
        )

    def reward(
        self,
        action: np.ndarray,
        next_state: np.ndarray,
    ) -> Tuple[float, bool]:
        """Reward function of walker.

        Args:
            action (np.ndarray): The action taken.
            next_state (np.ndarray): The next state.

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

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """ """
        # Send action to hub to receive next state
        action = tensordict.get("action").cpu().numpy().squeeze()
        self.send_to_hub(action)
        time.sleep(self.sleep_time)  # wait some time for sensors to read and to
        # receive the next state
        next_observation = self.read_from_hub()

        # calc reward and done
        reward, done = self.reward(
            action=action,
            next_state=next_observation,
        )
        next_tensordict = TensorDict(
            {
                self.observation_key: torch.tensor(
                    next_observation, dtype=torch.float32
                ),
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


def angular_difference(angle1, angle2):
    # Calculate the difference in angles, wrapped between -180 and 180
    difference = (angle2 - angle1 + 180) % 360 - 180
    return abs(difference)  # Return the absolute value of the difference
