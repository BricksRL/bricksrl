import time
from typing import Tuple

import numpy as np

import torch

from environments.base.base_env import BaseEnv
from tensordict import TensorDict, TensorDictBase
from torchrl.data.tensor_specs import BoundedTensorSpec, CompositeSpec


class BalanceEnv_v0(BaseEnv):
    """
    BalanceEnv_v0 is a custom gym environment for the 2wheeler balancing robot.

    Args:
        max_episode_steps (int): The maximum number of steps per episode. Defaults to 50.
        sleep_time (float): The time to wait between sending actions and receiving the next state. Defaults to 0.2.
        verbose (bool): Whether to print verbose information during the environment's execution. Defaults to False.

    Attributes:
        action_space (gym.spaces.Box): The continuous action space in the range [-1, 1].
        observation_space (gym.spaces.Box): The state space consisting of 5 sensor readings (left, right, pitch, roll, rotation_velocity) and 1 direction (left or right).

    Methods:
        sample_random_action() -> np.ndarray: Samples a random action from the action space.
        normalize_state(state: np.ndarray) -> np.ndarray: Normalizes and clips the state to be compatible with the agent.
        reset() -> np.ndarray: Resets the environment and returns the initial state.
    """

    action_dim = 4  # to control the wheel motors and the motor speed
    state_dim = 4  # 4 sensors (left, right, roll, rotation_velocity)

    motor_angles = (0, 360)
    roll_angles = (-90, 90)
    rotation_velocity = (-250, 250)  # adapt to real values
    observation_key = "vec_observation"

    def __init__(
        self,
        max_episode_steps: int = 50,
        sleep_time: float = 0.0,
        verbose: bool = False,
    ):
        self.sleep_time = sleep_time
        self._batch_size = torch.Size([1])
        self.max_episode_steps = max_episode_steps

        self.action_spec = BoundedTensorSpec(
            low=-torch.ones((1, self.action_dim)),
            high=torch.ones((1, self.action_dim)),
            shape=(1, self.action_dim),
        )

        observation_spec = BoundedTensorSpec(
            low=torch.tensor(
                [
                    [
                        self.motor_angles[0],
                        self.motor_angles[0],
                        self.roll_angles[0],
                        self.rotation_velocity[0],
                    ]
                ]
            ),
            high=torch.tensor(
                [
                    [
                        self.motor_angles[1],
                        self.motor_angles[1],
                        self.roll_angles[1],
                        self.rotation_velocity[1],
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
        norm_observation = self.normalize_state(observation)
        return TensorDict(
            {
                self.observation_key: norm_observation.float(),
                # "original_vec_observation": observation.squeeze(),
            },
            batch_size=[1],
        )

    def reward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Reward function of Spinning environment.
        If the self.direction is 0, the robot is spinning left, otherwise right.
        We want to maximise in those cases the angular velocity (last element of the state vector).
        If the robot is spinning in the wrong direction, we want to minimize the angular velocity.
        """
        # TODO: maybe add reward for low motor usage (energy efficiency) so that the robot relaxes when max distance is reached
        done = False
        velocity = tensordict.get((self.observation_key))[:, -1]
        roll = tensordict.get((self.observation_key))[:, -2]

        roll_reward = -np.abs(roll)

        # velocity_reward = - np.abs(velocity)/1000 # devide by 10 to put more focus on roll reward

        reward = roll_reward  # + velocity_reward
        tensordict.set("reward", torch.tensor(reward))
        tensordict.set("done", torch.tensor([done]))
        return tensordict

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """ """
        # Send action to hub to receive next state
        start_time = time.time()
        self.send_to_hub(tensordict.get("action").numpy().squeeze())
        time.sleep(
            self.sleep_time
        )  # we need to wait some time for sensors to read and to
        # receive the next state
        next_observation = self.read_from_hub()

        end_time = time.time()
        print(f"Time to send and receive: {(end_time - start_time):.2f} seconds")

        # calc reward and done
        next_tensordict = TensorDict(
            {
                self.observation_key: self.normalize_state(next_observation).float(),
            },
            batch_size=[1],
        )
        next_tensordict = self.reward(next_tensordict)

        # increment episode step counter
        self.episode_step_iter += 1
        if self.episode_step_iter >= self.max_episode_steps:
            next_tensordict.set("done", torch.tensor([True]))
            # To not act
            action = np.zeros(self.action_dim)
            self.send_to_hub(action)
            # do we need truncated?
        return next_tensordict
