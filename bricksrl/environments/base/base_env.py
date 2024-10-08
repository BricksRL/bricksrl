import struct
import sys

import numpy as np

import torch
from bricksrl.Pybricks.PybricksHubClass import PybricksHub
from tensordict import TensorDict, TensorDictBase
from torchrl.envs import EnvBase


class BaseEnv(EnvBase):
    """
    The base class for reinforcement learning environments used with the Lego robots.

    Args:
        action_dim (int): The dimensionality of the action space.
        state_dim (int): The dimensionality of the state space.
        use_hub (bool): Whether to use the Pybricks hub for communication, if False, only
            the observation spec and action specs are created and can be used.
            Can be helpful for testing and debugging as you dont connect to the hub.
        verbose (bool): Whether to print verbose output.
    """

    def __init__(
        self,
        action_dim: int,
        state_dim: int,
        use_hub: bool = True,
        verbose: bool = False,
    ):
        self.verbose = verbose
        self.action_dim = action_dim
        self.state_dim = state_dim

        self.action_format_str = "!" + "f" * self.action_dim
        self.state_format_str = "!" + "f" * self.state_dim

        self.expected_bytesize = struct.calcsize(self.state_format_str)

        # buffer state in case of missing data
        self.buffered_state = np.zeros(self.state_dim, dtype=np.float32)

        if use_hub:
            self.hub = PybricksHub(
                state_dim=state_dim, out_format_str=self.state_format_str
            )
            self.hub.connect()
            print("Connected to hub.")
        else:
            self.hub = None
        super().__init__(batch_size=torch.Size([1]))

    def send_to_hub(self, action: np.array) -> None:
        """
        Sends the given action to the hub as bytes.

        Args:
            action (np.array): The action to send to the hub as a numpy array.

        Raises:
            AssertionError: If the shape of the action does not match the action dimension.
        """
        assert (
            action.shape[0] == self.action_dim
        ), "Action shape does not match action dimension."
        byte_action = struct.pack(self.action_format_str, *action)
        if self.verbose:
            print("Sending data size: ", len(byte_action))
            print("Sending data: ", byte_action)
        self.hub.send(byte_action)

    def read_from_hub(self) -> np.array:
        """
        Reads the current state of the environment from the hub and returns it as a numpy array.

        Returns:
            np.array: The current state of the environment as a numpy array.
        """
        byte_state = self.hub.read()
        if self.verbose:
            print("Reading data size: ", sys.getsizeof(byte_state))
            print("Reading data: ", byte_state)
            print("len: ", len(byte_state))

        if len(byte_state) != self.expected_bytesize:
            print(
                "State has size {} but should have size {}.".format(
                    len(byte_state), struct.calcsize(self.state_format_str)
                )
            )
            print("Returning previous state.")
            state = self.buffered_state
            print("State: ", state)
        else:
            state = np.array([struct.unpack(self.state_format_str, byte_state)])
            self.buffered_state = state
        assert (
            state.shape[1] == self.state_dim
        ), f"State has shape {state.shape[0]} and does not match state dimension: {self.state_dim}."
        return state

    def sample_random_action(self, tensordict: TensorDictBase) -> TensorDictBase:
        """
        Sample a random action from the action space.

        Returns:
            TensorDictBase: A dictionary containing the sampled action.
        """
        if tensordict is not None:
            tensordict.set("action", self.action_spec.rand())
            return tensordict
        else:
            return TensorDict({"action": self.action_spec.rand()}, [])

    def close(self) -> None:
        if self.hub is not None:
            self.hub.close()

    def _step(
        self,
    ):
        raise NotImplementedError

    def _reset(
        self,
    ):
        raise NotImplementedError

    def _set_seed(self, seed: int):
        np.random.seed(seed)
        torch.manual_seed(seed)


class BaseSimEnv(EnvBase):
    """
    The base class for reinforcement learning environments used to simulate Lego robots.

    Args:
        action_dim (int): The dimensionality of the action space.
        state_dim (int): The dimensionality of the state space.
        verbose (bool): Whether to print verbose output.
        use_hub (bool): This argument is kept for compatibility but is not used in the simulation environment.
    """

    def __init__(
        self,
        action_dim: int,
        state_dim: int,
        verbose: bool = False,
        use_hub: bool = False,
    ):
        self.verbose = verbose
        self.action_dim = action_dim
        self.state_dim = state_dim

        super().__init__(batch_size=torch.Size([1]))

    def sample_random_action(self, tensordict: TensorDictBase) -> TensorDictBase:
        """
        Sample a random action from the action space.

        Returns:
            TensorDictBase: A dictionary containing the sampled action.
        """
        if tensordict is not None:
            tensordict.set("action", self.action_spec.rand())
            return tensordict
        else:
            return TensorDict({"action": self.action_spec.rand()}, [])

    def _step(
        self,
    ):
        raise NotImplementedError

    def _reset(
        self,
    ):
        raise NotImplementedError

    def _set_seed(self, seed: int):
        """
        Sets the seed for the environment's random number generator.

        Args:
            seed (int): The seed to set.
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
