import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.init as init
from tensordict import TensorDict, TensorDictBase
from torchrl.data.tensor_specs import TensorSpec
from torchrl.envs.utils import set_exploration_mode


class BaseAgent:
    """Implements a base agent used to interact with the lego robots.

    Args:
        state_space (gym.Space): The state space of the environment.
        action_space (gym.Space): The action space of the environment.
        device (torch.device): The device to use for computation.
        observation_keys (Tuple[str]): The keys used to access the observation in the tensor dictionary.

    Attributes:
        state_space (gym.Space): The state space of the environment.
        action_space (gym.Space): The action space of the environment.
        state_dim (int): The dimension of the state space.
        action_dim (int): The dimension of the action space.
        device (torch.device): The device to use for computation.
        observation_keys (Tuple[str]): The keys used to access the observation in the tensor dictionary.
    """

    def __init__(
        self,
        state_spec: TensorSpec,
        action_spec: TensorSpec,
        agent_name: str,
        device: str = "cpu",
    ):
        self.name = agent_name
        self.observation_spec = state_spec
        self.action_spec = action_spec
        self.device = device
        self.observation_keys = [key for key in self.observation_spec.keys()]

    def init_nets(self, model: nn.Module):
        """Initializes the networks with random data.

        Args:
            model (list): A list of PyTorch models to initialize.
        """
        with torch.no_grad(), set_exploration_mode("random"):
            td = self.observation_spec.rand()
            td = td.to(self.device)
            for net in model:
                net(td)
        del td

    @staticmethod
    def reset_parameter(param):
        if param.data.ndimension() == 2:  # Weights
            init.kaiming_uniform_(param.data, a=math.sqrt(5))
        else:  # Biases and others
            # Adjust based on your specific needs
            init.uniform_(param.data, -1, 1)

    def get_action(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Returns a sampled action given a tensordict to collect data.

        Args:
            tensordict (TensorDictBase): Tensordict containing the current state of the environment.

        Returns:
            TensorDictBase: TensorDict containing the sampled action to take in the environment.
        """
        raise NotImplementedError

    def get_eval_action(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Returns an action given a tensordict to evaluate the agent.

        Args:
            tensordict (TensorDictBase): Tensordict containing the current state of the environment.

        Returns:
            TensorDictBase: TensorDict containing the eval action to take in the environment.
        """
        raise NotImplementedError

    def train(
        self,
    ):
        """Trains the agent.

        Raises:
            NotImplementedError: This method must be implemented by a subclass.
        """
        raise NotImplementedError
