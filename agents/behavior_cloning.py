import numpy as np
import tensordict as td
import torch
from torch import nn, optim
from torchrl.data import (
    BoundedTensorSpec,
    TensorDictReplayBuffer,
)

from torchrl.data.replay_buffers.storages import LazyMemmapStorage

from agents.base import BaseAgent
from agents.networks import get_deterministic_actor, get_stochastic_actor


def initialize(net, std=0.02):
    for p, n in net.named_parameters():
        if "weight" in p:
            # nn.init.xavier_uniform_(n)
            nn.init.normal_(n, mean=0, std=std)
        elif "bias" in p:
            nn.init.zeros_(n)


class BehavioralCloningAgent(BaseAgent):
    def __init__(self, state_space, action_space, agent_config, device="cpu"):
        super(BehavioralCloningAgent, self).__init__(
            state_space, action_space, agent_config.name, device
        )

        # rewrite action spec to bounded tensor spec
        action_space = BoundedTensorSpec(
            minimum=torch.from_numpy(action_space.low).float(),
            maximum=torch.from_numpy(action_space.high).float(),
            shape=action_space.shape,
        )

        if agent_config.policy_type == "deterministic":
            self.actor = get_deterministic_actor(
                action_space,
                in_keys=["observation"],
                num_cells=[agent_config.num_cells, agent_config.num_cells],
                activation_class=nn.ReLU,
                normalization=agent_config.normalization,
                dropout=agent_config.dropout,

            )
            self.pretrain = self.pretrain_deter
        elif agent_config.policy_type == "stochastic":
            self.actor = get_stochastic_actor(
                action_space,
                in_keys=["observation"],
                num_cells=[agent_config.num_cells, agent_config.num_cells],
                activation_class=nn.ReLU,
                normalization=agent_config.normalization,
                dropout=agent_config.dropout,
            )
            self.pretrain = self.pretrain_stoch
        else:
            raise ValueError("policy_type not recognized, choose deterministic or stochastic")

        # initialize networks
        self.init_nets([self.actor])

        self.optimizer = optim.Adam(
            self.actor.parameters(), lr=agent_config.lr, weight_decay=0.0
        )

        # create replay buffer
        self.offline_data_path = agent_config.offline_data_path
        self.replay_buffer = self.create_replay_buffer()


        # general stats
        self.collected_transitions = 0
        self.do_pretrain = True
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

    def load_offline_data(self, path):
        """load offline data"""
        # TODO: cleanup!
        try:
            data = np.load(path, allow_pickle=True).item()
        except:
            raise ValueError("Cannot load offline data, check path!")
        
        return td.TensorDict(data, batch_size=len(data["observations"]))

        
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

        # load offline data
        if self.offline_data_path is not None:
            offline_data = self.load_offline_data(self.offline_data_path)
        
        replay_buffer.extend(offline_data)

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
            input_td = td.TensorDict({"observation": batch["observations"].float()}, batch_size=(256))
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
            pred, _ = self.actor(batch["observations"].float())
            loss = torch.mean((pred - batch["actions"]) ** 2)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            wandb.log({"pretrain/loss": loss.item()})

        self.actor.eval()


    def train(self, batch_size=64, num_updates=1):
        """Train the agent"""
        return {}