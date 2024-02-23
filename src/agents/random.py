import tensordict as td
import torch
from torchrl.data import BoundedTensorSpec

from src.agents.base import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self, state_spec, action_spec, agent_config, device="cpu"):
        super(RandomAgent, self).__init__(
            state_spec, action_spec, agent_config.name, device
        )

        self.actor = None
        self.do_pretrain = False
        self.replay_buffer = {}

    @torch.no_grad()
    def get_action(self, state):
        """Sample random actions from a uniform distribution"""
        return self.action_spec.rand().numpy()

    def add_experience(self, transition: td.TensorDict):
        """Add experience to replay buffer"""
        pass

    def train(self, batch_size=64, num_updates=1):
        """Train the agent"""
        return {}
