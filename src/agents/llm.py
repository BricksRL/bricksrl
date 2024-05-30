import torch
from openai import OpenAI
from tensordict import TensorDictBase

from src.agents.base import BaseAgent


class OpenAILLMAgent(BaseAgent):
    def __init__(self, state_spec, action_spec, agent_config, device="cpu"):
        super(OpenAILLMAgent, self).__init__(
            state_spec, action_spec, agent_config.name, device
        )

        with open(agent_config.openai_key, "r") as file:
            api_key = file.read().strip()
        self.client = OpenAI(api_key=api_key)
        self.model = agent_config.model
        self.actor = None
        self.replay_buffer = {}
        self.communication_history = []
        self.max_history_len = agent_config.max_history_len
        # load preprompt
        with open(agent_config.preprompt_dir, "r") as file:
            preprompt_text = file.read()

        self.communication_history.append({"role": "system", "content": preprompt_text})
        self.pending_actions = []
        self.num_actions = agent_config.num_actions

    def eval(self):
        """Sets the agent to evaluation mode."""

    # @torch.no_grad()
    # def get_action(self, tensordict: TensorDictBase):
    #     """Sample random actions from a uniform distribution"""

    #     observation = [tensordict.get(i) for i in self.observation_keys]
    #     # TODO: maybe we want to predict the next x actions instead of just one
    #     self.communication_history.append(
    #         {
    #             "role": "user",
    #             "content": f"The current observation is: {observation}. \nPlease provide the next action as a tuple (leftfront_value, leftback_value, rightfront_value, rightback_value), rounded to three decimal places. Only output the action tuple directly:",
    #         }
    #     )
    #     response = self.client.chat.completions.create(
    #         model=self.model,
    #         messages=self.communication_history,
    #     )

    #     content = response.choices[0].message.content
    #     # Add the assistant's response to the conversation history
    #     self.communication_history.append({"role": "assistant", "content": content})
    #     action_tuple = eval(content)
    #     if (
    #         type(action_tuple) == tuple
    #         and len(action_tuple) == self.action_spec.shape[-1]
    #     ):
    #         print(f"Action taken: {action_tuple}")
    #         tensordict.set("action", torch.tensor(action_tuple).unsqueeze(0))
    #     else:
    #         raise ValueError("The action must be a tuple of the correct shape")

    #     if len(self.communication_history) > self.max_history_len:
    #         del self.communication_history[1]

    #     return tensordict

    @torch.no_grad()
    def get_action(self, tensordict: TensorDictBase):
        """Sample the next x actions from the model"""
        if len(self.pending_actions) == 0:
            observation = [tensordict.get(i) for i in self.observation_keys]
            self.communication_history.append(
                {
                    "role": "user",
                    "content": f"The current observation is: {observation}. \nPlease provide the next {self.num_actions} sequential actions as a list of tuples (leftfront_value, leftback_value, rightfront_value, rightback_value), rounded to three decimal places. Only output the action tuples directly:",
                }
            )
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.communication_history,
            )
            content = response.choices[0].message.content
            # Add the assistant's response to the conversation history
            self.communication_history.append({"role": "assistant", "content": content})

            # Parse the response to extract actions
            self.pending_actions = eval(content)
            if not (
                isinstance(self.pending_actions, list)
                and all(
                    isinstance(action, tuple)
                    and len(action) == self.action_spec.shape[-1]
                    for action in self.pending_actions
                )
            ):
                raise ValueError(
                    "The actions must be a list of tuples of the correct shape"
                )

            if len(self.communication_history) > self.max_history_len:
                del self.communication_history[1]

        # Take the next action from the pending actions
        action_tuple = self.pending_actions.pop(0)
        print(f"Action taken: {action_tuple}")
        tensordict.set("action", torch.tensor(action_tuple).unsqueeze(0))

        return tensordict

    @torch.no_grad()
    def get_eval_action(self, tensordict: TensorDictBase):
        """Sample random actions from a uniform distribution"""
        tensordict.set("action", self.action_spec.rand())
        return tensordict

    def add_experience(self, transition: TensorDictBase):
        """Add experience to replay buffer"""
        pass

    def train(self, batch_size=64, num_updates=1):
        """Train the agent"""
        return {}
