import tensordict as td
import torch
from tensordict import TensorDictBase
from torch import optim
from torchrl.data import TensorDictPrioritizedReplayBuffer, TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyMemmapStorage, LazyTensorStorage
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives import SoftUpdate

from torchrl.objectives.cql import CQLLoss

from src.agents.base import BaseAgent
from src.networks.networks import get_critic, get_stochastic_actor


class CQLAgent(BaseAgent):
    def __init__(self, state_spec, action_spec, agent_config, device="cpu"):
        super(CQLAgent, self).__init__(
            state_spec, action_spec, agent_config.name, device
        )

        with_lagrange = agent_config.with_lagrange

        self.actor = get_stochastic_actor(
            state_spec, action_spec, agent_config
        )
        self.critic = get_critic(state_spec, agent_config)

        self.actor.to(device)
        self.critic.to(device)

        # initialize networks
        self.init_nets([self.actor, self.critic])

        # define loss function
        self.loss_module = CQLLoss(
            actor_network=self.actor,
            qvalue_network=self.critic,
            loss_function=agent_config.loss_function,
            temperature=agent_config.temperature,
            min_q_weight=agent_config.min_q_weight,
            max_q_backup=agent_config.max_q_backup,
            deterministic_backup=agent_config.deterministic_backup,
            num_random=agent_config.num_random,
            with_lagrange=agent_config.with_lagrange,
            lagrange_thresh=agent_config.lagrange_thresh,
        )
        # Define Target Network Updater
        self.target_net_updater = SoftUpdate(
            self.loss_module, eps=agent_config.soft_update_eps
        )
        self.target_net_updater.init_()

        # Reset weights
        self.reset_params = agent_config.reset_params

        # Define Replay Buffer
        self.batch_size = agent_config.batch_size
        self.replay_buffer = self.create_replay_buffer(
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
        if with_lagrange:
            self.alpha_prime_optim = torch.optim.Adam(
                [self.loss_module.log_alpha_prime],
                lr=agent_config.lr,
            )
        else:
            self.alpha_prime_optim = None
        # general stats
        self.collected_transitions = 0
        self.total_updates = 0
        self.do_pretrain = agent_config.pretrain
        self.bc_steps = agent_config.bc_steps

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

    def eval(self):
        """Sets the agent to evaluation mode."""
        self.actor.eval()

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
        # replay_buffer.append_transform(ToTensorImage(from_int=True, shape_tolerant=True))
        return replay_buffer

    @torch.no_grad()
    def get_action(self, td: TensorDictBase) -> TensorDictBase:
        """Get action from actor network"""
        with set_exploration_type(ExplorationType.RANDOM):
            out_td = self.actor(td.to(self.device))
        self.td_preprocessing(out_td)
        return out_td

    @torch.no_grad()
    def get_eval_action(self, td: TensorDictBase) -> TensorDictBase:
        """Get eval action from actor network"""
        with set_exploration_type(ExplorationType.MODE):
            out_td = self.actor(td.to(self.device))
        self.td_preprocessing(out_td)
        return out_td

    def add_experience(self, transition: td.TensorDict):
        """Add experience to replay buffer"""
        self.replay_buffer.extend(transition)
        self.collected_transitions += 1

    def train(self, batch_size=64, num_updates=1):
        """Train the agent"""
        self.actor.train()
        for i in range(num_updates):
            self.total_updates += 1
            # Sample a batch from the replay buffer
            batch = self.replay_buffer.sample(batch_size)
            # Compute CQL Loss
            loss = self.loss_module(batch)

            # Update alpha
            alpha_loss = loss["loss_alpha"]
            alpha_prime_loss = loss["loss_alpha_prime"]
            self.optimizer_alpha.zero_grad()
            alpha_loss.backward()
            self.optimizer_alpha.step()

            # Update Actpr Network
            # official cql implementation uses behavior cloning loss for first few updating steps as it helps for some tasks
            if self.total_updates >= self.bc_steps:
                actor_loss = loss["loss_actor"]
            else:
                actor_loss = loss["loss_actor_bc"]
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()

            if self.alpha_prime_optim is not None:
                self.alpha_prime_optim.zero_grad()
                alpha_prime_loss.backward(retain_graph=True)
                self.alpha_prime_optim.step()

            # Update Critic Network
            q_loss = loss["loss_qvalue"]
            cql_loss = loss["loss_cql"]

            q_loss = q_loss + cql_loss
            self.optimizer_critic.zero_grad()
            q_loss.backward(retain_graph=False)
            self.optimizer_critic.step()

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
