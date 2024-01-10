import tensordict as td
import torch
from torchrl.data import (
    BoundedTensorSpec,
)

from agents.base import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self, state_space, action_space, agent_config, device="cpu"):
        super(RandomAgent, self).__init__(
            state_space, action_space, agent_config.name, device
        )

        # rewrite action spec to bounded tensor spec
        self.action_space = BoundedTensorSpec(
            minimum=torch.from_numpy(action_space.low).float(),
            maximum=torch.from_numpy(action_space.high).float(),
            shape=action_space.shape,
        )
        self.actor = None
        self.do_pretrain = False
        self.replay_buffer = {}

    @torch.no_grad()
    def get_action(self, state):
        """Sample random actions from a uniform distribution"""
        return self.action_space.rand().numpy()

    def add_experience(self, transition: td.TensorDict):
        """Add experience to replay buffer"""
        pass
        

    def train(self, batch_size=64, num_updates=1):
        """Train the agent"""
        return {}