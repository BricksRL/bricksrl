import time
from typing import Tuple

import numpy as np
import torch
from environments.base.base_env import BaseEnv
from tensordict import TensorDict, TensorDictBase
from torchrl.data.tensor_specs import BoundedTensorSpec, CompositeSpec


class RunAwayEnv_v0(BaseEnv):
    """
    A reinforcement learning environment for training agents to get away from a wall.

    The goal of the agent is to increase the distance measured by an ultrasonic sensor and get away from the wall as fast as possible.
    The environment provides a state consisting of 4 sensor readings (left, right, pitch, roll) and the distance to the wall.
    The agent can take a continuous action in the range [-1, 1] to control the movement of the robot.
    The environment returns a reward based on the change in distance to the wall and terminates the episode if the robot gets too close to the wall or the maximum number of steps is reached.

    Args:
        max_episode_steps (int): The maximum number of steps per episode. Defaults to 10.
        min_distance (float): The minimum distance to the wall. Defaults to 40.
        sleep_time (float): The time to wait between sending actions and receiving the next state. Defaults to 0.2.
        verbose (bool): Whether to print verbose information during the environment's execution. Defaults to False.

    """

    action_dim = 1  # control the wheel motors together
    # 5 sensors (left motor angle, right motor angle, pitch, roll, distance)
    state_dim = 5

    observation_ranges = {
        "left_motor_angles": [0, 360],
        "right_motor_angles": [0, 360],
        "roll_angle": [-90, 90],
        "pitch_angle": [-90, 90],
        "distance": [0, 2000],
    }

    observation_key = "observation"

    def __init__(
        self,
        max_episode_steps: int = 10,
        min_distance: float = 40,
        sleep_time: float = 0.2,
        verbose: bool = False,
        pretrain: bool = False,
    ):
        self.sleep_time = sleep_time
        self.min_distance = min_distance
        self.max_episode_steps = max_episode_steps
        self._batch_size = torch.Size([1])

        # Define action spec
        self.action_spec = BoundedTensorSpec(
            low=-1,
            high=1,
            shape=(1, self.action_dim),
        )

        # Define observation spec
        bounds = torch.tensor(
            [
                self.observation_ranges["left_motor_angles"],
                self.observation_ranges["right_motor_angles"],
                self.observation_ranges["roll_angle"],
                self.observation_ranges["pitch_angle"],
                self.observation_ranges["distance"],
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
        self.verbose = verbose
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
            action = tensordict.get("action").cpu().numpy().squeeze(0)
        else:
            action = np.zeros(self.action_dim)
        self.send_to_hub(action)
        time.sleep(self.sleep_time)
        observation = self.read_from_hub()
        self.distance = observation[:, -1]
        return TensorDict(
            {
                self.observation_key: torch.tensor(observation, dtype=torch.float32),
                "distance": torch.tensor([self.distance]).float(),
            },
            batch_size=[1],
        )

    def reward(self, next_observation: np.array) -> Tuple[float, bool]:
        """Reward function of RunAwayEnv.

        Goal: Increase distance measured by ultrasonic sensor aka.
        get away from the wall as fast as possible.

        """
        done = False

        current_distance = next_observation[:, -1]
        if current_distance <= self.min_distance:  # too close to the wall break episode
            done = True
            reward = 0.0
        elif current_distance < self.distance:
            reward = -1.0
        elif current_distance > self.distance:
            reward = 1.0
        else:
            reward = 0.0
        if self.distance >= 2000:
            done = True
        self.distance = current_distance
        return reward, done

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """ """
        # Send action to hub to receive next state
        self.send_to_hub(tensordict.get("action").cpu().numpy().squeeze(0))
        time.sleep(self.sleep_time)  # wait some time for sensors to read and to

        # receive the next state
        next_observation = self.read_from_hub()

        # calc reward and done
        reward, done = self.reward(
            next_observation=next_observation,
        )

        next_tensordict = TensorDict(
            {
                self.observation_key: torch.tensor(
                    next_observation, dtype=torch.float32
                ),
                "reward": torch.tensor([reward]).float(),
                "done": torch.tensor([done]).bool(),
                "distance": torch.tensor([self.distance]).float(),
            },
            batch_size=[1],
        )

        # increment episode step counter
        self.episode_step_iter += 1
        if self.episode_step_iter >= self.max_episode_steps:
            next_tensordict.set("done", torch.tensor([True]))
        return next_tensordict
