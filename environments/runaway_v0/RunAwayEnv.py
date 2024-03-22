import time

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

    Attributes:
        action_space (gym.spaces.Box): The continuous action space in the range [-1, 1].
        observation_space (gym.spaces.Box): The state space consisting of 4 sensor readings and the distance to the wall.

    Methods:
        sample_random_action(): Samples a random action from the action space.
        normalize_state(state: np.ndarray) -> np.ndarray: Normalizes and clips the state to be compatible with the agent.
        reset() -> np.ndarray: Resets the environment and returns the initial state.
        reward(state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> Tuple[float, bool]: Calculates the reward based on the change in distance to the wall.
        step(action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]: Performs the given action and returns the next state, reward, and done status.
    """

    action_dim = 1  # control the wheel motors together
    state_dim = (
        5  # 5 sensors (left motor angle, right motor angle, pitch, roll, distance)
    )
    motor_angles = (0, 360)
    roll_angles = (-90, 90)
    pitch_angles = (-90, 90)
    distance = (0, 2000)
    observation_key = "vec_observation"
    original_vec_observation_key = "original_vec_observation"

    def __init__(
        self,
        max_episode_steps: int = 10,
        min_distance: float = 40,
        sleep_time: float = 0.2,
        verbose: bool = False,
    ):
        self.sleep_time = sleep_time
        self.min_distance = min_distance
        self.max_episode_steps = max_episode_steps
        self._batch_size = torch.Size([1])

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
                        self.pitch_angles[0],
                        self.distance[0],
                    ]
                ]
            ),
            high=np.array(
                [
                    [
                        self.motor_angles[1],
                        self.motor_angles[1],
                        self.roll_angles[1],
                        self.pitch_angles[1],
                        self.distance[1],
                    ]
                ]
            ),
        )
        self.observation_spec = CompositeSpec(shape=(1,))
        self.observation_spec.set(self.observation_key, observation_spec)
        self.verbose = verbose
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
            action = tensordict.get("action").cpu().numpy().squeeze(0)
        else:
            action = np.zeros(self.action_dim)
        self.send_to_hub(action)
        time.sleep(self.sleep_time)
        observation = self.read_from_hub()
        norm_observation = self.normalize_state(observation)
        return TensorDict(
            {
                self.observation_key: norm_observation.float(),
                self.original_vec_observation_key: torch.from_numpy(
                    observation
                ).float(),  # for reward calc
            },
            batch_size=[1],
        )

    def reward(self, state, next_state) -> TensorDictBase:
        """Reward function of RunAwayEnv.

        Goal: Increase distance measured by ultrasonic sensor aka.
        get away from the wall as fast as possible.

        """
        done = False

        if (
            next_state[:, -1] <= self.min_distance
        ):  # too close to the wall break episode
            reward = -10.0
            done = True
        elif next_state[:, -1] < state[:, -1]:
            reward = -1.0
        elif next_state[:, -1] > state[:, -1]:
            reward = 1.0
        else:
            reward = 0.0
        if state[:, -1] >= 2000:
            done = True
        return reward, done

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """ """
        # Send action to hub to receive next state
        self.send_to_hub(tensordict.get("action").cpu().numpy().squeeze(0))
        time.sleep(
            self.sleep_time
        )  # we need to wait some time for sensors to read and to
        # receive the next state
        next_observation = self.read_from_hub()
        next_tensordict = TensorDict(
            {
                self.observation_key: self.normalize_state(next_observation).float(),
                self.original_vec_observation_key: torch.from_numpy(
                    next_observation
                ).float(),  # for reward calc
            },
            batch_size=[1],
        )
        # calc reward and done
        reward, done = self.reward(
            state=tensordict.get(self.original_vec_observation_key),
            next_state=next_tensordict.get(self.original_vec_observation_key),
        )
        next_tensordict.set("reward", torch.tensor([reward]).float())
        next_tensordict.set("done", torch.tensor([done]).bool())

        # increment episode step counter
        self.episode_step_iter += 1
        if self.episode_step_iter >= self.max_episode_steps:
            next_tensordict.set("done", torch.tensor([True]))
        return next_tensordict
