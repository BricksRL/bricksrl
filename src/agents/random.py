import torch
from tensordict import TensorDictBase

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
    def get_action(self, tensordict: TensorDictBase):
        """Sample random actions from a uniform distribution"""
        tensordict.set("action", self.action_spec.rand())
        return tensordict

    @torch.no_grad()
    def get_eval_action(self, tensordict: TensorDictBase):
        """Sample random actions from a uniform distribution"""
        tensordict.set("action", self.action_spec.rand())
        return tensordict

    def add_experience(self, transition: TensorDictBase):
        """Add experience to replay buffer"""
        pass

    def train(self, batch_size=64, num_updates=1):
        """Train the agent"""
        return {}
