import numpy as np
from environments.base.PybricksHubClass import PybricksHub
import struct

class BaseEnv():
    """ Base class for all environments to communicate with the Pybricks Hub."""
    def __init__(self, action_dim: int, state_dim: int,):
        self.action_dim = action_dim
        self.state_dim = state_dim
        
        self.action_format_str = "!" + "f" * self.action_dim
        self.state_format_str = "!" + "f" * self.state_dim

        self.hub = PybricksHub(state_dim=state_dim, out_format_str=self.state_format_str)
        self.hub.connect()
        print("Connected to hub.")

    def send_to_hub(self, action: np.array)-> None:
        """ Takes action as numpy array and sends it to the hub as bytes. """
        assert action.shape[0] == self.action_dim, "Action shape does not match action dimension."
        byte_action = struct.pack(self.action_format_str, action)
        self.hub.send(byte_action)
        
    def read_from_hub(self)-> np.array:
        """ Reads state from the hub as bytes and converts it to numpy array. """
        byte_state = self.hub.read()
        state = np.array([struct.unpack(self.state_format_str, byte_state)])
        assert state.shape[1] == self.state_dim, f"State has shape {state.shape[0]} and does not match state dimension: {self.state_dim}."
        return state
    
    def _step(self,):
        raise NotImplementedError
    
    def _reset(self,):
        raise NotImplementedError
    
    def _reward(self,):
        raise NotImplementedError
    
    
        

