import numpy as np
import tensordict as td
import torch
from tensordict import TensorDictBase
from torch import nn, optim
from torchrl.data import BoundedTensorSpec, TensorDictReplayBuffer

from torchrl.data.replay_buffers.storages import LazyMemmapStorage
from torchrl.envs.utils import ExplorationType, set_exploration_type

from src.agents.base import BaseAgent
from src.networks.networks import get_deterministic_actor, get_stochastic_actor


def initialize(net, std=0.02):
    for p, n in net.named_parameters():
        if "weight" in p:
            # nn.init.xavier_uniform_(n)
            nn.init.normal_(n, mean=0, std=std)
        elif "bias" in p:
            nn.init.zeros_(n)


class BehavioralCloningAgent(BaseAgent):
    def __init__(self, state_spec, action_spec, agent_config, device="cpu"):
        super(BehavioralCloningAgent, self).__init__(
            state_spec, action_spec, agent_config.name, device
        )

        if agent_config.policy_type == "deterministic":
            self.actor = get_deterministic_actor(
                self.observation_keys, action_spec, agent_config
            )
            self.pretrain = self.pretrain_deter
        elif agent_config.policy_type == "stochastic":
            self.actor = get_stochastic_actor(
                self.observation_keys, action_spec, agent_config
            )
            self.pretrain = self.pretrain_stoch
        else:
            raise ValueError(
                "policy_type not recognized, choose deterministic or stochastic"
            )
        self.actor.to(device)
        # initialize networks
        self.init_nets([self.actor])

        self.optimizer = optim.Adam(
            self.actor.parameters(), lr=agent_config.lr, weight_decay=0.0
        )

        # create replay buffer
        self.offline_data_path = None
        self.replay_buffer = self.create_replay_buffer()

        # general stats
        self.collected_transitions = 0
        self.do_pretrain = False
        self.episodes = 0

    def get_agent_statedict(self):
        """Save agent"""
        act_statedict = self.actor.state_dict()
        return {"actor": act_statedict}

    def load_model(self, path):
        """load model"""
        try:
            statedict = torch.load(path)
            self.actor.load_state_dict(statedict["actor"])
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

    def eval(self):
        """Sets the agent to evaluation mode."""
        self.actor.eval()

    @torch.no_grad()
    def get_eval_action(self, td: TensorDictBase) -> TensorDictBase:
        """Get eval action from actor network"""
        with set_exploration_type(ExplorationType.MODE):
            out_td = self.actor(td.to(self.device))
        return out_td

    def create_replay_buffer(
        self,
        batch_size=256,
        buffer_size=10000,
        buffer_scratch_dir=None,
        device="cpu",
        prefetch=3,
    ):
        """Create replay buffer"""

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
        # set exploration mode?
        out_td = self.actor(input_td).squeeze(0)
        return out_td["action"].cpu().numpy()

    def add_experience(self, transition: td.TensorDict):
        """Add experience to replay buffer"""

        # TODO: for bc we dont want to add to replay buffer
        pass

    def pretrain_stoch(self, wandb, batch_size=64, num_updates=1):
        """Pretrain the agent with simple behavioral cloning"""

        for i in range(num_updates):
            batch = self.replay_buffer.sample(batch_size)
            input_td = td.TensorDict(
                {"observation": batch["vec_observations"].float()}, batch_size=(256)
            )
            dist = self.actor.get_dist(input_td)
            loss = -dist.log_prob(batch["actions"]).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            wandb.log({"pretrain/loss": loss.item()})

        self.actor.eval()

    def pretrain_deter(self, wandb, batch_size=64, num_updates=1):
        """Pretrain the agent with simple behavioral cloning"""

        for i in range(num_updates):
            batch = self.replay_buffer.sample(batch_size)
            pred, _ = self.actor(batch["vec_observations"].float())
            loss = torch.mean((pred - batch["actions"]) ** 2)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            wandb.log({"pretrain/loss": loss.item()})

        self.actor.eval()

    def train(self, batch_size=64, num_updates=1):
        """Train the agent"""
        log_data = {}

        for i in range(num_updates):
            batch = self.replay_buffer.sample(batch_size).to(self.device)
            orig_action = batch.get("action").clone()
            out_dict = self.actor(batch)
            loss = torch.mean((out_dict.get("action") - orig_action) ** 2)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            log_data.update({"loss": loss})
        return log_data
