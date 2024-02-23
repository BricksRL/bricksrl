import numpy as np
import tensordict as td
import torch
from torch import nn, optim
from torchrl.data import (
    BoundedTensorSpec,
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.data.replay_buffers.storages import LazyMemmapStorage, LazyTensorStorage
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives import SoftUpdate

from torchrl.objectives.sac import SACLoss

from agents.base import BaseAgent
from agents.networks import get_critic, get_stochastic_actor
from typing import Tuple
from tensordict import TensorDictBase


class SACAgent(BaseAgent):
    def __init__(self, state_space, action_spec, agent_config, device="cpu"):
        super(SACAgent, self).__init__(
            state_space, action_spec, agent_config.name, device
        )
        
        self.actor = get_stochastic_actor(self.observation_keys, action_spec, agent_config)
        self.critic = get_critic(self.observation_keys, agent_config)

        # initialize networks
        self.init_nets([self.actor, self.critic])

        # set initial network weights
        # use a small std to start with small action values at the beginning
        # initialize(self.actor, std=0.02)

        # define loss function
        self.loss_module = SACLoss(
            actor_network=self.actor,
            qvalue_network=self.critic,
            delay_qvalue=True,
            value_network=None,  # None to use SAC version 2
            num_qvalue_nets=2,
            gamma=agent_config.gamma,
            fixed_alpha=agent_config.fixed_alpha,
            alpha_init=agent_config.alpha_init,
            loss_function=agent_config.loss_function,
        )
        # Define Target Network Updater
        self.target_net_updater = SoftUpdate(
            self.loss_module, eps=agent_config.soft_update_eps
        )
        self.target_net_updater.init_()

        # Define Replay Buffer
        self.replay_buffer = self.create_replay_buffer(
            batch_size=agent_config.batch_size,
            prb=agent_config.prb,
            buffer_size=agent_config.buffer_size,
            device=device,
        )

        # Define Optimizer
        critic_params = list(
            self.loss_module.qvalue_network_params.flatten_keys().values()
        )
        actor_params = list(
            self.loss_module.actor_network_params.flatten_keys().values()
        )
        self.optimizer_actor = optim.Adam(
            actor_params, lr=agent_config.lr, weight_decay=0.0
        )
        self.optimizer_critic = optim.Adam(
            critic_params, lr=agent_config.lr, weight_decay=0.0
        )
        self.optimizer_alpha = optim.Adam(
            [self.loss_module.log_alpha],
            lr=3.0e-4,
        )

        # general stats
        self.collected_transitions = 0
        self.episodes = 0
        self.do_pretrain = agent_config.pretrain

    def get_agent_statedict(self):
        """Save agent"""
        act_statedict = self.actor.state_dict()
        critic_statedict = self.critic.state_dict()
        return {"actor": act_statedict, "critic": critic_statedict}

    def load_model(self, path):
        """load model"""
        try:
            statedict = torch.load(path)
            self.actor.load_state_dict(statedict["actor"])
            self.critic.load_state_dict(statedict["critic"])
            print("Model loaded")
        except:
            raise ValueError("Model not loaded")

    def load_replaybuffer(self, path):
        """load replay buffer"""
        try:
            self.replay_buffer.load_state_dict(torch.load(path))
            print("Replay Buffer loaded")
            print("Replay Buffer size: ", self.replay_buffer.__len__(), "\n")
        except:
            raise ValueError("Replay Buffer not loaded")

    def td_preprocessing(self, td: TensorDictBase) -> TensorDictBase:
        # TODO not ideal to have this here
        td.pop("scale")
        td.pop("loc")
        td.pop("params")
        if "vector_obs_embedding" in td.keys():
            td.pop("vector_obs_embedding")
        if "image_embedding" in td.keys():
            td.pop("image_embedding")


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

    @torch.no_grad()
    def get_action(self, td: TensorDictBase) -> TensorDictBase:
        """Get action from actor network"""
        with set_exploration_type(ExplorationType.RANDOM):
            out_td = self.actor(td)
        self.td_preprocessing(out_td)
        return out_td

    def add_experience(self, transition: td.TensorDict):
        """Add experience to replay buffer"""
        self.replay_buffer.extend(transition)
        self.collected_transitions += 1

    def pretrain(self, wandb, batch_size=64, num_updates=1):
        """Pretrain the agent with simple behavioral cloning"""
        # TODO: implement pretrain for testing
        # for i in range(num_updates):
        #     batch = self.replay_buffer.sample(batch_size)
        #     pred, _ = self.actor(batch["observations"].float())
        #     loss = torch.mean((pred - batch["actions"]) ** 2)
        #     self.optimizer.zero_grad()
        #     loss.backward()
        #     self.optimizer.step()
        #     wandb.log({"pretrain/loss": loss.item()})

    def train(self, batch_size=64, num_updates=1):
        """Train the agent"""
        self.actor.train()
        for i in range(num_updates):
            # Sample a batch from the replay buffer
            batch = self.replay_buffer.sample(batch_size)
            # Compute SAC Loss
            loss = self.loss_module(batch)

            # Update Actpr Network
            self.optimizer_actor.zero_grad()
            loss["loss_actor"].backward()
            self.optimizer_actor.step()
            # Update Critic Network
            self.optimizer_critic.zero_grad()
            loss["loss_qvalue"].backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.optimizer_critic.step()

            # Update alpha
            self.optimizer_alpha.zero_grad()
            loss["loss_alpha"].backward()
            self.optimizer_alpha.step()

            # Update Target Networks
            self.target_net_updater.step()
            # Update Prioritized Replay Buffer
            if isinstance(self.replay_buffer, TensorDictPrioritizedReplayBuffer):
                self.replay_buffer.update_priorities(
                    batch["indices"],
                    loss["critic_loss"].detach().cpu().numpy(),
                )
        self.actor.eval()
        return loss
