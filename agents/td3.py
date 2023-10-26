import numpy as np
import tensordict as td
import torch
from torch import nn, optim
from torchrl.data import (
    BoundedTensorSpec,
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.modules import AdditiveGaussianWrapper
from torchrl.objectives import SoftUpdate
from torchrl.objectives.td3 import TD3Loss

from agents.base import BaseAgent
from agents.networks import get_critic, get_deterministic_actor


def initialize(net, std=0.02):
    for p, n in net.named_parameters():
        if "weight" in p:
            # nn.init.xavier_uniform_(n)
            nn.init.normal_(n, mean=0, std=std)
        elif "bias" in p:
            nn.init.zeros_(n)


class TD3Agent(BaseAgent):
    def __init__(self, state_space, action_space, agent_config, device="cpu"):
        super(TD3Agent, self).__init__(state_space, action_space, device)

        # rewrite action spec to bounded tensor spec
        action_space = BoundedTensorSpec(
            minimum=torch.from_numpy(action_space.low).float(),
            maximum=torch.from_numpy(action_space.high).float(),
            shape=action_space.shape,
        )
        self.actor = get_deterministic_actor(
            action_space,
            in_keys=["observation"],
            num_cells=[agent_config.num_cells, agent_config.num_cells],
            activation_class=nn.ReLU,
        )
        self.critic = get_critic(
            in_keys=["observation"],
            out_features=1,
            num_cells=[agent_config.num_cells, agent_config.num_cells],
            activation_class=nn.ReLU,
        )

        self.actor_explore = AdditiveGaussianWrapper(
            self.actor,
            sigma_init=1,
            sigma_end=1,
            mean=0,
            std=0.01,
        ).to(device)
        # initialize networks
        self.init_nets([self.actor, self.critic])

        # set initial network weights
        # use a small std to start with small action values at the beginning
        # initialize(self.actor, std=0.02)

        # define loss function
        self.loss_module = TD3Loss(
            actor_network=self.actor,
            qvalue_network=self.critic,
            num_qvalue_nets=2,
            gamma=agent_config.gamma,
            loss_function=agent_config.loss_function,
        )
        # Define Target Network Updater
        self.target_net_updater = SoftUpdate(
            self.loss_module, agent_config.soft_update_eps
        )
        self.target_net_updater.init_()

        # Define Replay Buffer
        self.replay_buffer = self.create_replay_buffer(
            prb=False, buffer_size=100_000, device=device
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

        # general stats
        self.collected_transitions = 0
        self.episodes = 0

    def get_agent_statedict(self):
        """Save agent"""
        act_statedict = self.actor.state_dict()
        critic_statedict = self.critic.state_dict()
        return {"actor": act_statedict, "critic": critic_statedict}

    def load_replaybuffer(self, path):
        """load replay buffer"""
        self.replay_buffer.load_state_dict(torch.load(path))
        print("Replay Buffer loaded")
        print("Replay Buffer size: ", self.replay_buffer.__len__(), "\n")

    def create_replay_buffer(self, prb=False, buffer_size=100000, device="cpu"):
        if prb:
            replay_buffer = TensorDictPrioritizedReplayBuffer(
                alpha=0.7,
                beta=0.5,
                pin_memory=False,
                prefetch=3,
                storage=LazyTensorStorage(
                    buffer_size,
                    device=device,
                ),
            )
        else:
            replay_buffer = TensorDictReplayBuffer(
                pin_memory=False,
                prefetch=3,
                storage=LazyTensorStorage(
                    buffer_size,
                    device=device,
                ),
            )
        return replay_buffer

    @torch.no_grad()
    def get_action(self, state):
        """Get action from actor network"""

        state = torch.from_numpy(state).float().to(self.device)[None, :]
        input_td = td.TensorDict({"observation": state}, batch_size=1)

        out_td = self.actor_explore(input_td).squeeze(0)
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
            # Compute TD3 Loss
            loss = self.loss_module(batch)
            # Update Critic Network
            self.optimizer_critic.zero_grad()
            loss["loss_qvalue"].backward(retain_graph=True)
            self.optimizer_critic.step()
            # Update Actor Network
            if i % 2 == 0:
                self.optimizer_actor.zero_grad()
                loss["loss_actor"].backward()
                self.optimizer_actor.step()
                # Update Target Networks
                self.target_net_updater.step()
            # Update Prioritized Replay Buffer
            if isinstance(self.replay_buffer, TensorDictPrioritizedReplayBuffer):
                self.replay_buffer.update_priorities(
                    batch["indices"],
                    loss["critic_loss"].detach().cpu().numpy(),
                )
        return loss
