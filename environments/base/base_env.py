import numpy as np
from environments.base.PybricksHubClass import PybricksHub
import struct
import sys
from gym import Env

class BaseEnv(Env):
    """ Base class for all environments to communicate with the Pybricks Hub."""
    def __init__(self, action_dim: int, state_dim: int,):
        self.action_dim = action_dim
        self.state_dim = state_dim
        
        self.action_format_str = "!" + "f" * self.action_dim
        self.state_format_str = "!" + "f" * self.state_dim
        
        self.expected_bytesize = struct.calcsize(self.state_format_str)
        
        # buffer state in case of missing data
        self.buffered_state = np.zeros(self.state_dim, dtype=np.float32)

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
        print("Reading data size: ", sys.getsizeof(byte_state))
        print("Reading data: ", byte_state)
        print("len: ", len(byte_state))
        # assert sys.getsizeof(byte_state) == 53, f"State has size {sys.getsizeof(byte_state)} but should have size 53."
        if len(byte_state) != self.expected_bytesize:
            print("State has size {} but should have size {}.".format(len(byte_state), struct.calcsize(self.state_format_str)))
            print("Returning previous state.")
            state = self.buffered_state
            print("State: ", state)
        else:
            state = np.array([struct.unpack(self.state_format_str, byte_state)])
            self.buffered_state = state
        assert state.shape[1] == self.state_dim, f"State has shape {state.shape[0]} and does not match state dimension: {self.state_dim}."
        return state
    
    def close(self)-> None:
        self.hub.close()
    
    def _step(self,):
        raise NotImplementedError
    
    def _reset(self,):
        raise NotImplementedError
    
    def _reward(self,):
        raise NotImplementedError
    
    def render(self, ):
        raise NotImplementedError
    
    
        

