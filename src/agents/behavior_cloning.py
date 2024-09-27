import numpy as np
import tensordict as td
import torch
from tensordict import TensorDictBase
from torch import nn, optim
from torchrl.data import BoundedTensorSpec, TensorDictReplayBuffer

from torchrl.data.replay_buffers.storages import LazyMemmapStorage
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.envs import RenameTransform, ToTensorImage

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
                state_spec, action_spec, agent_config
            )
        elif agent_config.policy_type == "stochastic":
            self.actor = get_stochastic_actor(
                state_spec, action_spec, agent_config
            )
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
        self.batch_size = agent_config.batch_size
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
        buffer_size=1000000,
        buffer_scratch_dir="./tmp",
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
            ),
            batch_size=self.batch_size,
        )
        replay_buffer.append_transform(lambda x: x.to(device))
        # TODO: check if we have image in observation space if so add this transform
        #replay_buffer.append_transform(ToTensorImage(from_int=True, shape_tolerant=True))

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

        pass

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
