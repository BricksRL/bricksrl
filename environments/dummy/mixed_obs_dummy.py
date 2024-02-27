from typing import Tuple

import numpy as np

import torch

from tensordict import TensorDict, TensorDictBase
from torchrl.data.tensor_specs import BoundedTensorSpec, CompositeSpec
from torchrl.envs import EnvBase


class MixedObsDummyEnv(EnvBase):
    """
    MixedObsDummyEnv is a dummy environment for testing purposes.
    It does not connec to PyBricks

    """

    action_dim = 4
    state_dim = 7
    vec_observation_key = "vec_observation"
    image_observation_key = "image_observation"

    def __init__(
        self,
    ):
        self._batch_size = torch.Size([1])
        self.action_spec = BoundedTensorSpec(
            low=-torch.ones((1, self.action_dim)),
            high=torch.ones((1, self.action_dim)),
            shape=(1, self.action_dim),
        )

        observation_spec = BoundedTensorSpec(
            low=-torch.ones((1, self.state_dim)),
            high=torch.ones((1, self.state_dim)),
        )

        image_observation_spec = BoundedTensorSpec(
            low=torch.zeros((1,) + (64, 64, 3), dtype=torch.uint8),
            high=torch.ones((1,) + (64, 64, 3), dtype=torch.uint8) * 255,
        )

        self.observation_spec = CompositeSpec(shape=(1,))
        self.observation_spec.set(self.vec_observation_key, observation_spec)
        self.observation_spec.set(self.image_observation_key, image_observation_spec)
        super().__init__(batch_size=self._batch_size)

    def _set_seed(self, seed: int):
        return super()._set_seed(seed)

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        """
        Reset the environment and return the initial state.

        Returns:
            TensorDictBase: The initial state of the environment.
        """
        # TODO solve this fake action sending before to receive first state
        self.episode_step_iter = 0
        observation = self.observation_spec[self.vec_observation_key].rand()
        image_observation = self.observation_spec[self.image_observation_key].rand()
        return TensorDict(
            {
                self.vec_observation_key: observation.float(),
                self.image_observation_key: image_observation,
            },
            batch_size=[1],
        )

    def reward(
        self,
        action: np.ndarray,
        next_state: np.ndarray,
    ) -> Tuple[float, bool]:
        """ """
        return 0.0, False

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """ """
        action = tensordict.get("action").numpy()
        observation = self.observation_spec[self.vec_observation_key].rand()
        image_observation = self.observation_spec[self.image_observation_key].rand()

        reward, done = self.reward(
            action=action,
            next_state=observation,
        )
        next_tensordict = TensorDict(
            {
                self.vec_observation_key: observation.float(),
                self.image_observation_key: image_observation,
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
