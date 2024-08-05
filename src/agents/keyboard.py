import time

import tensordict as td
import torch
from pynput import keyboard
from tensordict import TensorDictBase
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyMemmapStorage

from src.agents.base import BaseAgent


class KeyboardAgent(BaseAgent):
    def __init__(self, state_spec, action_spec, agent_config, device="cpu"):
        super(KeyboardAgent, self).__init__(
            state_spec, action_spec, agent_config.name, device
        )

        # Define the key to action mapping
        self.key_action_mapping = {
            "a": [0.15, 0, 0, 0],  # Rotate motor -30
            "d": [-0.15, 0, 0, 0],  # Rotate motor +30
            "s": [0, -0.20, 0, 0],  # Low motor -10
            "w": [0, 0.20, 0, 0],  # Low motor +10
            "q": [0, 0, -0.25, 0],  # High motor -15
            "e": [0, 0, 0.25, 0],  # High motor +15
            "f": [0, 0, 0, -0.25],  # Grab motor -10
            "g": [0, 0, 0, 0.25],  # Grab motor +10
        }
        self.current_action = None
        self.setup_key_listener()
        self.buffer_batch_size = agent_config.batch_size
        # Define Replay Buffer
        self.replay_buffer = self.create_replay_buffer(
            batch_size=self.buffer_batch_size,
            prb=False,
            buffer_size=agent_config.buffer_size,
            device=device,
        )

        # general stats
        self.collected_transitions = 0
        self.total_updates = 0

    def setup_key_listener(self):
        def on_press(key):
            try:
                if key.char in self.key_action_mapping:
                    self.current_action = self.key_action_mapping[key.char]
            except AttributeError:
                pass

        def on_release(key):
            self.current_action = None

        self.listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.listener.start()

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
            # self.replay_buffer.load(path)
            loaded_data = TensorDictBase.load_memmap(path).to_tensordict()
            self.replay_buffer.extend(loaded_data)
            if self.replay_buffer._batch_size != self.buffer_batch_size:
                Warning(
                    "Batch size of the loaded replay buffer is different from the agent's config batch size! Rewriting the batch size to match the agent's config batch size."
                )
                self.replay_buffer._batch_size = self.buffer_batch_size
            print("Replay Buffer loaded")
            print("Replay Buffer size: ", self.replay_buffer.__len__(), "\n")
        except:
            raise ValueError("Replay Buffer not loaded")

    def eval(self):
        """Sets the agent to evaluation mode."""
        pass

    def create_replay_buffer(
        self,
        batch_size=256,
        prb=False,
        buffer_size=100000,
        buffer_scratch_dir="./scratch",
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
            batch_size=batch_size,
        )
        replay_buffer.append_transform(lambda x: x.to(device))



        return replay_buffer

    @torch.no_grad()
    def get_action(self, td: TensorDictBase) -> TensorDictBase:
        """Get action from actor network or keyboard"""
        while self.current_action is None:
            time.sleep(0.01)  # Add a small sleep to avoid blocking
        td.set("action", torch.tensor(self.current_action).float().unsqueeze(0))
        return td

    @torch.no_grad()
    def get_eval_action(self, td: TensorDictBase) -> TensorDictBase:
        """Get action from actor network or keyboard"""
        while self.current_action is None:
            time.sleep(0.01)  # Add a small sleep to avoid blocking
        td.set("action", torch.tensor(self.current_action).float().unsqueeze(0))
        return td

    def add_experience(self, transition: td.TensorDict):
        """Add experience to replay buffer"""
        
        # transform pixels to int if pixels in observation spec
        save_transition = transition.copy()
        if "pixels" in transition.keys():
            save_transition.set("pixels", (save_transition.get("pixels")*255).to(torch.int64))
            save_transition.set(("next", "pixels"), (save_transition.get(("next", "pixels"))*255).to(torch.int64))

        self.replay_buffer.extend(save_transition)
        self.collected_transitions += 1

    def train(self, batch_size=64, num_updates=1):
        """Train the agent"""
        return {}
