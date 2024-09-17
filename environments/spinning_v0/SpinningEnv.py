import time
from typing import Tuple

import numpy as np
import torch
from environments.base.base_env import BaseEnv
from tensordict import TensorDict, TensorDictBase
from torchrl.data.tensor_specs import BoundedTensorSpec, CompositeSpec


class SpinningEnv_v0(BaseEnv):
    """
    SpinningEnv_v0 is a custom gym environment for a spinning robot.
    The robot has to learn to spin in a circle around its own axis given a random goal direction (left or right, 0 or 1).

    Args:
        max_episode_steps (int): The maximum number of steps per episode. Defaults to 50.
        sleep_time (float): The time to wait between sending actions and receiving the next state. Defaults to 0.2.
        verbose (bool): Whether to print verbose information during the environment's execution. Defaults to False.

    """

    action_dim = 2  # to control the wheel motors independently
    state_dim = 5  # 5 sensors (left, right, pitch, roll, rotation_velocity) + 1 direction (left or right)

    observation_ranges = {
        "left_motor_angle": [0, 360],
        "right_motor_angle": [0, 360],
        "pitch_angle": [-90, 90],
        "roll_angle": [-90, 90],
        "rotation_velocity": [-100, 100],
        "direction": [0, 1],
    }

    observation_key = "observation"

    def __init__(
        self,
        max_episode_steps: int = 50,
        sleep_time: float = 0.2,
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
                self.observation_ranges["left_motor_angle"],
                self.observation_ranges["right_motor_angle"],
                self.observation_ranges["pitch_angle"],
                self.observation_ranges["roll_angle"],
                self.observation_ranges["rotation_velocity"],
                self.observation_ranges["direction"],
            ]
        )
        low_bounds = bounds[:, 0].unsqueeze(0)
        high_bounds = bounds[:, 1].unsqueeze(0)

        observation_spec = BoundedTensorSpec(
            low=low_bounds,
            high=high_bounds,
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

        state = self.read_from_hub()
        self.direction = np.random.randint(0, 2)  # (0,1) left or right
        full_original_state = np.concatenate(
            (state, np.array([[self.direction]])), axis=1, dtype=np.float32
        )

        return TensorDict(
            {
                self.observation_key: torch.tensor(full_original_state),
            },
            batch_size=[1],
        )

    def reward(self, next_observation: np.array) -> Tuple[float, bool]:
        """Reward function of Spinning environment.
        If the self.direction is 0, the robot is spinning left, otherwise right.
        We want to maximise in those cases the angular velocity (last element of the state vector).
        If the robot is spinning in the wrong direction, we want to minimize the angular velocity.
        """
        done = False
        velocity = next_observation[:, -2]

        if self.direction == 0:
            reward = velocity
        else:
            reward = -velocity

        return reward, done

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """ """
        # Send action to hub to receive next state
        self.send_to_hub(tensordict.get("action").cpu().numpy().squeeze())
        time.sleep(self.sleep_time)  # wait some time for sensors to read and to
        # receive the next state
        next_observation = self.read_from_hub()
        full_original_next_observation = np.concatenate(
            (next_observation, np.array([[self.direction]])), axis=1, dtype=np.float32
        )
        # calc reward and done
        reward, done = self.reward(full_original_next_observation)

        next_tensordict = TensorDict(
            {
                self.observation_key: torch.tensor(full_original_next_observation),
                "reward": torch.tensor([reward]).float(),
                "done": torch.tensor([done]).bool(),
            },
            batch_size=[1],
        )
        # increment episode step counter
        self.episode_step_iter += 1
        if self.episode_step_iter >= self.max_episode_steps:
            next_tensordict.set("done", torch.tensor([True]).bool())

        return next_tensordict
