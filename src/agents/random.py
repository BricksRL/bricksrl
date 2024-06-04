import torch
from tensordict import TensorDictBase
from torchrl.data import TensorDictPrioritizedReplayBuffer, TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyMemmapStorage, LazyTensorStorage

from src.agents.base import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self, state_spec, action_spec, agent_config, device="cpu"):
        super(RandomAgent, self).__init__(
            state_spec, action_spec, agent_config.name, device
        )

        self.actor = None
        self.do_pretrain = False
        self.replay_buffer = self.create_replay_buffer(
            batch_size=agent_config.batch_size,
            prb=False,
            buffer_size=100000,
            device=device,
        )

    def create_replay_buffer(
        self,
        batch_size=256,
        prb=False,
        buffer_size=100000,
        buffer_scratch_dir=None,
        device="cpu",
        prefetch=3,
    ):
        """Create replay buffer"""
        # TODO: make this part of base off policy agent
        if prb:
            replay_buffer = TensorDictPrioritizedReplayBuffer(
                alpha=0.7,
                beta=0.5,
                pin_memory=False,
                prefetch=1,
                storage=LazyTensorStorage(
                    buffer_size,
                    device=device,
                ),
            )
        else:
            replay_buffer = TensorDictReplayBuffer(
                pin_memory=False,
                prefetch=prefetch,
                storage=LazyMemmapStorage(
                    buffer_size,
                    scratch_dir=buffer_scratch_dir,
                    device=device,
                ),
                batch_size=batch_size,
            )
        return replay_buffer

    def eval(self):
        """Sets the agent to evaluation mode."""

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
        self.replay_buffer.extend(transition)

    def train(self, batch_size=64, num_updates=1):
        """Train the agent"""
        return {}
