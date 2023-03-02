import numpy as np
import torch
from torch import nn, optim
from torchrl.data import TensorDictPrioritizedReplayBuffer, TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyMemmapStorage
from torchrl.objectives import SoftUpdate
from torchrl.objectives.sac import SACLoss

from agents.base import BaseAgent
from agents.networks import get_stochastic_actor, get_critic
import tensordict as td

def initialize(net, std=0.02):
    for p, n in net.named_parameters():
        if "weight" in p:
            # nn.init.xavier_uniform_(n)
            nn.init.normal_(n, mean=0, std=std)
        elif "bias" in p:
            nn.init.zeros_(n)

class SACAgent(BaseAgent):
    def __init__(self, state_space, action_space, learning_rate=3e-4, device="cpu"):
        super(SACAgent, self).__init__(state_space, action_space, device)

    
        self.actor = get_stochastic_actor(action_space, in_keys=["observation"], num_cells=[256, 256], activation_class=nn.ReLU)
        self.critic = get_critic(in_keys=["observation"], out_features=1, num_cells=[256, 256], activation_class=nn.ReLU)

        # initialize networks
        self.init_nets([self.actor, self.critic])
        
        # set initial network weights
        # use a small std to start with small action values at the beginning
        # initialize(self.actor, std=0.02)
        
        # define loss function
        self.loss_module = SACLoss(actor_network=self.actor,
                                   qvalue_network=self.critic,
                                   delay_qvalue=True,
                                   value_network=None, # None to use SAC version 2
                                   num_qvalue_nets=2,
                                   gamma=0.99,
                                   loss_function="smooth_l1")
        # Define Target Network Updater
        self.target_net_updater = SoftUpdate(self.loss_module, 0.995)
        self.target_net_updater.init_()
        
        # Define Replay Buffer
        self.replay_buffer = self.create_replay_buffer(prb=False, buffer_size=100_000, buffer_scratch_dir="/tmp/", device=device)

        # Define Optimizer
        critic_params = list(self.loss_module.qvalue_network_params.flatten_keys().values())
        actor_params = list(self.loss_module.actor_network_params.flatten_keys().values())
        self.optimizer_actor = optim.Adam(actor_params, lr=learning_rate, weight_decay=0.0)
        self.optimizer_critic = optim.Adam(critic_params, lr=learning_rate, weight_decay=0.0)
        
        # general stats
        self.collected_transitions = 0
        self.episodes = 0

    def create_replay_buffer(self, prb=False, buffer_size=100000, buffer_scratch_dir="/tmp/", device="cpu",make_replay_buffer=3):
        
        if prb:
            replay_buffer = TensorDictPrioritizedReplayBuffer(
                alpha=0.7,
                beta=0.5,
                pin_memory=False,
                prefetch=make_replay_buffer,
                storage=LazyMemmapStorage(
                    buffer_size,
                    scratch_dir=buffer_scratch_dir,
                    device=device,
                ),
            )
        else:
            replay_buffer = TensorDictReplayBuffer(
                pin_memory=False,
                prefetch=make_replay_buffer,
                storage=LazyMemmapStorage(
                    buffer_size,
                    scratch_dir=buffer_scratch_dir,
                    device=device,
                ),
            )
        return replay_buffer
    
    @torch.no_grad()
    def get_action(self, state):
        """Get action from actor network"""

        state = torch.from_numpy(state).float().to(self.device)
        input_td = td.TensorDict({"observation": state}, batch_size=1)
        # set exploration mode?
        out_td = self.actor(input_td).squeeze(0)
        return out_td["action"].cpu().numpy()
    
    def add_experience(self, transition: td.TensorDict):
        """Add experience to replay buffer"""
        self.replay_buffer.extend(transition)
        self.collected_transitions += 1
    
    def train(self, batch_size=64, num_updates=1):
        """Train the agent"""
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
            self.optimizer_critic.step()
            # Update Target Networks
            self.target_net_updater.step()
            # Update Prioritized Replay Buffer
            if isinstance(self.replay_buffer, TensorDictPrioritizedReplayBuffer):
                self.replay_buffer.update_priorities(
                    batch["indices"],
                    loss["critic_loss"].detach().cpu().numpy(),
                )
        return loss