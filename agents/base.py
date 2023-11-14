import torch
from torchrl.envs.utils import set_exploration_mode
from tensordict.tensordict import TensorDict

class BaseAgent():
    """ Implements a base agent used to interact with the lego robots.

    Args:
        state_space (gym.Space): The state space of the environment.
        action_space (gym.Space): The action space of the environment.
        device (torch.device): The device to use for computation.

    Attributes:
        state_space (gym.Space): The state space of the environment.
        action_space (gym.Space): The action space of the environment.
        state_dim (int): The dimension of the state space.
        action_dim (int): The dimension of the action space.
        device (torch.device): The device to use for computation.
    """
    def __init__(self, state_space, action_space, agent_name, device):
        
        self.name = agent_name
        self.state_space = state_space
        self.action_space = action_space
        self.state_dim = state_space.shape[0]
        self.action_dim = action_space.shape[0]
        
        self.device = device
    
    def get_dummy_td(self):
        """ Returns a dummy tensor dict for testing purposes.

        Returns:
            TensorDict: A dictionary of tensors representing a batch of data.
        """
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
        """ Initializes the networks with random data.

        Args:
            model (list): A list of PyTorch models to initialize.
        """
        with torch.no_grad(), set_exploration_mode("random"):
            td = self.get_dummy_td()
            td = td.to(self.device)
            for net in model:
                net(td)
        del td
    
    def get_action(self, state):
        """ Returns an action given a state.

        Args:
            state (np.ndarray): The current state of the environment.

        Returns:
            np.ndarray: The action to take in the environment.
        """
        raise NotImplementedError
    
    def train(self,):
        """ Trains the agent.

        Raises:
            NotImplementedError: This method must be implemented by a subclass.
        """
        raise NotImplementedError
    