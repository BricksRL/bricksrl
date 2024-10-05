import tensordict as td
import torch

from experiments.helper.agents.base import BaseAgent
from experiments.helper.networks.networks import get_critic, get_deterministic_actor

from tensordict import TensorDictBase
from torch import nn, optim
from torchrl.data import TensorDictPrioritizedReplayBuffer, TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyMemmapStorage, LazyTensorStorage

from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import AdditiveGaussianWrapper
from torchrl.objectives import SoftUpdate
from torchrl.objectives.td3 import TD3Loss
from torchrl.objectives.td3_bc import TD3BCLoss


def initialize(net, std=0.02):
    for p, n in net.named_parameters():
        if "weight" in p:
            # nn.init.xavier_uniform_(n)
            nn.init.normal_(n, mean=0, std=std)
        elif "bias" in p:
            nn.init.zeros_(n)


class TD3Agent(BaseAgent):
    def __init__(self, state_spec, action_spec, agent_config, device="cpu"):
        super(TD3Agent, self).__init__(
            state_spec, action_spec, agent_config.name, device
        )

        self.actor = get_deterministic_actor(state_spec, action_spec, agent_config)
        self.critic = get_critic(state_spec, agent_config)

        self.model = nn.ModuleList([self.actor, self.critic]).to(device)

        print(self.actor)
        print(self.critic)
        # initialize networks
        self.init_nets(self.model)

        self.actor_explore = AdditiveGaussianWrapper(
            self.model[0],
            sigma_init=1,
            sigma_end=1,
            mean=0,
            std=agent_config.exploration_noise,
        ).to(device)

        # define loss function
        self.use_bc = agent_config.use_bc
        if not self.use_bc:
            self.loss_module = TD3Loss(
                actor_network=self.model[0],
                qvalue_network=self.model[1],
                action_spec=action_spec,
                num_qvalue_nets=2,
                loss_function=agent_config.loss_function,
                separate_losses=False,
            )
        else:
            self.loss_module = TD3BCLoss(
                actor_network=self.model[0],
                qvalue_network=self.model[1],
                action_spec=action_spec,
                num_qvalue_nets=2,
                loss_function=agent_config.loss_function,
                separate_losses=False,
                alpha=agent_config.alpha,
            )

        # Define Target Network Updater
        self.target_net_updater = SoftUpdate(
            self.loss_module, eps=agent_config.soft_update_eps
        )
        self.target_net_updater.init_()

        self.batch_size = agent_config.batch_size
        # Define Replay Buffer
        self.replay_buffer = self.create_replay_buffer(
            prb=agent_config.prb,
            buffer_size=agent_config.buffer_size,
            device=device,
            buffer_scratch_dir="/tmp",
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

        # Reset weights
        self.reset_params = agent_config.reset_params
        # general stats
        self.collected_transitions = 0
        # td stats for delayed update
        self.total_updates = 0

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
            loaded_data = TensorDictBase.load_memmap(path)
            self.replay_buffer.extend(loaded_data)
            if self.replay_buffer._batch_size != self.batch_size:
                Warning(
                    "Batch size of the loaded replay buffer is different from the agent's config batch size! Rewriting the batch size to match the agent's config batch size."
                )
                self.replay_buffer._batch_size = self.batch_size
            print("Replay Buffer loaded")
            print("Replay Buffer size: ", self.replay_buffer.__len__(), "\n")
        except:
            raise ValueError("Replay Buffer not loaded")

    def reset_networks(self):
        """reset network parameters"""
        print("Resetting Networks!")
        self.loss_module.actor_network_params.apply(self.reset_parameter)
        self.loss_module.target_actor_network_params.apply(self.reset_parameter)
        self.loss_module.qvalue_network_params.apply(self.reset_parameter)
        self.loss_module.target_qvalue_network_params.apply(self.reset_parameter)

    def create_replay_buffer(
        self,
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
                ),
            )
        else:
            replay_buffer = TensorDictReplayBuffer(
                pin_memory=False,
                prefetch=prefetch,
                storage=LazyMemmapStorage(
                    buffer_size,
                    scratch_dir=buffer_scratch_dir,
                ),
                batch_size=self.batch_size,
            )
        replay_buffer.append_transform(lambda x: x.to(device))
        # TODO: check if we have image in observation space if so add this transform
        # replay_buffer.append_transform(
        #     ToTensorImage(
        #         from_int=True,
        #         shape_tolerant=True,
        #         in_keys=["pixels", ("next", "pixels")],
        #     )
        # )
        return replay_buffer

    def td_preprocessing(self, td: TensorDictBase) -> TensorDictBase:
        # TODO not ideal to have this here
        td.pop("param")
        if "obs_embedding" in td.keys():
            td.pop("obs_embedding")
        if "pixel_embedding" in td.keys():
            td.pop("pixel_embedding")

    def eval(self):
        """Sets the agent to evaluation mode."""
        self.actor.eval()

    @torch.no_grad()
    def get_action(self, td: TensorDictBase) -> TensorDictBase:
        """Get action from actor network"""
        with set_exploration_type(ExplorationType.RANDOM):
            out_td = self.actor_explore(td.to(self.device))
        self.actor_explore.step(1)
        self.td_preprocessing(out_td)
        return out_td

    @torch.no_grad()
    def get_eval_action(self, td: TensorDictBase) -> TensorDictBase:
        """Get eval action from actor network"""
        with set_exploration_type(ExplorationType.MODE):
            out_td = self.actor(td.to(self.device))
        # self.td_preprocessing(out_td)
        return out_td

    def add_experience(self, transition: td.TensorDict):
        """Add experience to replay buffer"""
        self.replay_buffer.extend(transition)
        self.collected_transitions += 1

    def train(self, batch_size=64, num_updates=1):
        """Train the agent"""
        log_data = {}
        for _ in range(num_updates):
            self.total_updates += 1
            if self.reset_params and self.total_updates % self.reset_params == 0:
                self.reset_networks()
            # Sample a batch from the replay buffer
            sampled_tensordict = self.replay_buffer.sample()
            if sampled_tensordict.device != self.device:
                sampled_tensordict = sampled_tensordict.to(
                    self.device, non_blocking=True
                )
            else:
                sampled_tensordict = sampled_tensordict.clone()
            # Update Critic Network
            if self.use_bc:
                q_loss, _ = self.loss_module.qvalue_loss(sampled_tensordict)
            else:
                q_loss, _ = self.loss_module.value_loss(sampled_tensordict)
            self.optimizer_critic.zero_grad()
            q_loss.backward()
            self.optimizer_critic.step()
            log_data.update({"critic_loss": q_loss})

            # Update Actor Network
            if self.total_updates % 2 == 0:
                actor_loss, _ = self.loss_module.actor_loss(sampled_tensordict)
                self.optimizer_actor.zero_grad()
                actor_loss.backward()
                self.optimizer_actor.step()

                # Update Target Networks
                self.target_net_updater.step()
                log_data.update({"actor_loss": actor_loss})

            # Update Prioritized Replay Buffer
            if isinstance(self.replay_buffer, TensorDictPrioritizedReplayBuffer):
                self.replay_buffer.update_priorities(sampled_tensordict)

        return log_data
