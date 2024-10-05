from typing import Tuple

import numpy as np

import torch

from bricksrl.environments.base.base_env import BaseEnv
from tensordict import TensorDict, TensorDictBase
from torchrl.data.tensor_specs import BoundedTensorSpec, CompositeSpec


class CustomEnv(BaseEnv):
    """
    Environment template for creating your own custom environment for BricksRL.

    Args:
        max_episode_steps (int): The maximum number of steps per episode. Defaults to 10.
        verbose (bool): Whether to print additional information. Defaults to False.

    """

    def __init__(
        self,
        max_episode_steps: int = 50,
        verbose: bool = False,
    ):
        self._batch_size = torch.Size([1])
        self.max_episode_steps = max_episode_steps

        # Define action spec
        self.action_spec = BoundedTensorSpec(
            low=-1,
            high=1,
            shape=(1, self.action_dim),
        )

        # Define observation spec

        observation_spec = BoundedTensorSpec(
            low=-1,
            high=1,
            shape=(1, self.state_dim),
        )

        self.observation_spec = CompositeSpec(
            {self.observation_key: observation_spec}, shape=(1,)
        )
        super().__init__(
            action_dim=self.action_dim,
            state_dim=self.state_dim,
            verbose=verbose,
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
        # Get current observation
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
        """Your custom reward function"""
        return 1.0, False

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Custom step function"""
        # Send action to hub to receive next state
        action = tensordict.get("action").cpu().numpy().squeeze()
        self.send_to_hub(action)
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
