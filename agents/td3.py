import copy 
import tensordict as td
import torch
from torch import nn, optim
from torchrl.data import (
    BoundedTensorSpec,
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.data.replay_buffers.storages import LazyMemmapStorage, LazyTensorStorage
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
        super(TD3Agent, self).__init__(
            state_space, action_space, agent_config.name, device
        )

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
            # normalization=agent_config.normalization,
            # dropout=agent_config.dropout,
        )
        self.critic = get_critic(
            in_keys=["observation"],
            out_features=1,
            num_cells=[agent_config.num_cells, agent_config.num_cells],
            activation_class=nn.ReLU,
            normalization=agent_config.normalization,
            dropout=agent_config.dropout,
        )
        # initialize networks
        self.init_nets([self.actor, self.critic])

        self.actor_explore = AdditiveGaussianWrapper(
            self.actor,
            sigma_init=1,
            sigma_end=1,
            mean=0,
            std=0.01,
        ).to(device)

        # set initial network weights
        # use a small std to start with small action values at the beginning
        # initialize(self.actor, std=0.02)

        # define loss function
        self.loss_module = TD3Loss(
            actor_network=self.actor,
            qvalue_network=self.critic,
            action_spec=action_space,
            num_qvalue_nets=2,
            gamma=agent_config.gamma,
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

        # general stats
        self.collected_transitions = 0
        self.episodes = 0
        # td stats for delayed update
        self.total_updates = 0
        self.do_pretrain = False

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
    def get_action(self, state):
        """Get action from actor network"""

        state = torch.from_numpy(state).float().to(self.device)[None, :]
        input_td = td.TensorDict({"observation": state}, batch_size=1)

        out_td = self.actor(input_td).squeeze(0)
        self.actor_explore.step(1)
        return out_td["action"].cpu().numpy()

    def add_experience(self, transition: td.TensorDict):
        """Add experience to replay buffer"""
        self.replay_buffer.extend(transition)
        self.collected_transitions += 1

    def train(self, batch_size=64, num_updates=1):
        """Train the agent"""
        log_data = {}
        for i in range(num_updates):
            self.total_updates += 1
            # Sample a batch from the replay buffer
            sampled_tensordict = self.replay_buffer.sample()
            if sampled_tensordict.device != self.device:
                sampled_tensordict = sampled_tensordict.to(
                    self.device, non_blocking=True
                )
            else:
                sampled_tensordict = sampled_tensordict.clone()
            # out = self.loss_module(sampled_tensordict)
            # q_loss = out["loss_qvalue"]
            # actor_loss = out["loss_actor"]
            # Update Critic Network
            q_loss, *_ = self.loss_module.value_loss(sampled_tensordict)
            self.optimizer_critic.zero_grad()
            q_loss.backward()
            self.optimizer_critic.step()
            log_data.update({"critic_loss": q_loss.item()})

            # Update Actor Network
            if self.total_updates % 2 == 0:
                actor_loss, *_ = self.loss_module.actor_loss(sampled_tensordict)
                self.optimizer_actor.zero_grad()
                actor_loss.backward()
                self.optimizer_actor.step()

                # Update Target Networks
                self.target_net_updater.step()
                log_data.update({"actor_loss": actor_loss.item()})

            # Update Prioritized Replay Buffer
            if isinstance(self.replay_buffer, TensorDictPrioritizedReplayBuffer):
                self.replay_buffer.update_priorities(sampled_tensordict)
            

        return log_data
