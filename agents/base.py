import torch
from torchrl.envs.utils import set_exploration_mode
from tensordict.tensordict import TensorDict

class BaseAgent():
    """ Implements a base agent used to interact with the lego robots"""
    def __init__(self, state_space, action_space, device):
        
        self.state_space = state_space
        self.action_space = action_space
        self.state_dim = state_space.shape[0]
        self.action_dim = action_space.shape[0]
        
        self.device = device
    
    def get_dummy_td(self):
        """ Returns a dummy tensor dict"""
        batch=8

        # create a tensordict
        obs = torch.randn(batch, self.state_dim, device=self.device)
        next_obs = torch.randn(batch, self.state_dim, device=self.device)

        action = torch.randn(batch, self.action_dim, device=self.device).clamp(-1, 1)
        reward = torch.randn(batch, 1, device=self.device)
        done = torch.zeros(batch, 1, dtype=torch.bool, device=self.device)
        td = TensorDict(
            batch_size=(batch,),
            source={
                "observation": obs,
                "next": {"observation": next_obs},
                "done": done,
                "reward": reward,
                "action": action,
            },
            device=self.device,
        )
        return td

    def init_nets(self, model):
        """ Initializes the networks"""
        with torch.no_grad(), set_exploration_mode("random"):
            td = self.get_dummy_td()
            td = td.to(self.device)
            for net in model:
                net(td)
        del td
    
    def get_action(self, state):
        """ Returns an action given a state"""
        raise NotImplementedError
    
    def train(self,):
        """ Trains the agent"""
        raise NotImplementedError
    