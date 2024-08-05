import tensordict as td
import torch
from tensordict import TensorDictBase
from torch import optim
from torchrl.data import TensorDictPrioritizedReplayBuffer, TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyMemmapStorage, LazyTensorStorage
from torchrl.envs.transforms import ToTensorImage
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives import SoftUpdate

from torchrl.objectives.iql import IQLLoss

from src.agents.base import BaseAgent
from src.networks.networks import get_critic, get_stochastic_actor, get_value_operator


class IQLAgent(BaseAgent):
    def __init__(self, state_spec, action_spec, agent_config, device="cpu"):
        super(IQLAgent, self).__init__(
            state_spec, action_spec, agent_config.name, device
        )

        self.actor = get_stochastic_actor(
            self.observation_keys, action_spec, agent_config
        )
        self.critic = get_critic(self.observation_keys, agent_config)

        self.value = get_value_operator(self.observation_keys, agent_config)

        self.actor.to(device)
        self.critic.to(device)
        self.value.to(device)

        # initialize networks
        self.init_nets([self.actor, self.critic, self.value])

        # define loss function
        self.loss_module = IQLLoss(
            actor_network=self.actor,
            qvalue_network=self.critic,
            value_network=self.value,
            num_qvalue_nets=2,
            temperature=agent_config.temperature,
            expectile=agent_config.expectile,
            loss_function=agent_config.loss_function,
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
            batch_size=self.batch_size,
            prb=agent_config.prb,
            buffer_size=agent_config.buffer_size,
            device=device,
        )

        # Define Optimizer
        critic_params = list(
            self.loss_module.qvalue_network_params.flatten_keys().values()
        )
        value_params = list(
            self.loss_module.value_network_params.flatten_keys().values()
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
        self.optimizer_value = optim.Adam(
            value_params, lr=agent_config.lr, weight_decay=0.0
        )

        # general stats
        self.collected_transitions = 0
        self.total_updates = 0
        self.do_pretrain = agent_config.pretrain

    def get_agent_statedict(self):
        """Save agent"""
        act_statedict = self.actor.state_dict()
        critic_statedict = self.critic.state_dict()
        value_statedict = self.value.state_dict()
        return {
            "actor": act_statedict,
            "critic": critic_statedict,
            "value": value_statedict,
        }

    def load_model(self, path):
        """load model"""

        try:
            statedict = torch.load(path)
            self.actor.load_state_dict(statedict["actor"])
            self.critic.load_state_dict(statedict["critic"])
            self.value.load_state_dict(statedict["value"])
            print("Model loaded")
        except:
            raise ValueError("Model not loaded")

    def load_replaybuffer(self, path):
        """load replay buffer"""
        try:
            # self.replay_buffer.load(path)
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
        self.loss_module.value_network_params.apply(self.reset_parameter)

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
                ),
                batch_size=batch_size,
            )
        replay_buffer.append_transform(lambda x: x.to(device))
        replay_buffer.append_transform(
            ToTensorImage(
                from_int=True,
                shape_tolerant=True,
                in_keys=["pixels", ("next", "pixels")],
            )
        )

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
            self.total_updates += 1
            if self.reset_params and self.total_updates % self.reset_params == 0:
                self.reset_networks()
            # Sample a batch from the replay buffer
            batch = self.replay_buffer.sample(batch_size)
            # Compute IQL Loss
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
            # Update Value Network
            self.optimizer_value.zero_grad()
            loss["loss_value"].backward()
            self.optimizer_value.step()

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
