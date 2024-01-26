import numpy as np
import torch
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.objectives.value.utils import _get_num_per_traj, _split_and_pad_sequence


class HERSampling:
    def __init__(
        self,
        observation_space,
        generation_type,
        reward_signal="dense",
        goal_thresholds=20,
        samples=4,
    ):
        self.observation_space = observation_space
        self.generation_type = generation_type
        self.samples = samples
        self.goal_thresholds = goal_thresholds
        self.transition_buffer = []
        if reward_signal == "dense":
            self.reward_function = dense_reward_function
        elif reward_signal == "sparse":
            self.reward_function = sparse_reward_function
        else:
            raise ValueError("Invalid reward type")

    def add_transition(self, td, info):
        """adds transitions to the buffer to wait until one epoch is full to do the sampling"""

        td.set("desired_state", torch.from_numpy(info["desired_state"]).unsqueeze(0))
        td.set("achieved_state", torch.from_numpy(info["achieved_state"]).unsqueeze(0))
        self.transition_buffer.append(td)
        if td.get("next")["truncated"] == True or td.get("next")["done"] == True:
            trajectory = torch.stack(self.transition_buffer).squeeze()
            augmentation_td = self.her_augmentation(trajectory.clone())
            b, t = augmentation_td.shape
            self.transition_buffer = []
            return augmentation_td.reshape(b * t)
        else:
            return None

    def generate_future_idxs(self, traj_lens):
        def generate_for_single_traj_len(traj_len):
            idxs = []
            for i in range(traj_len - 1):
                idxs.append(
                    torch.randint(low=i + 1, high=traj_len, size=(1, self.samples))
                )
            # correct for the last idx with last idx
            idxs.append(torch.full((1, self.samples), fill_value=traj_len - 1))
            return torch.cat(idxs)

        return [generate_for_single_traj_len(traj_len) for traj_len in traj_lens]

    def generate_random_idxs(self, traj_lens):
        def generate_for_single_traj_len(traj_len):
            idxs = []
            for _ in range(traj_len):
                idxs.append(torch.randint(low=0, high=traj_len, size=(1, self.samples)))
            return torch.cat(idxs)

        return [generate_for_single_traj_len(traj_len) for traj_len in traj_lens]

    def generate_final_idx(self, traj_lens):
        def generate_for_single_traj_len(traj_len):
            return torch.full((traj_len, self.samples), fill_value=traj_len - 1)

        return [generate_for_single_traj_len(traj_len) for traj_len in traj_lens]

    def generate_sample_idxs(self, trajectories):
        if self.generation_type == "future":
            idxs = self.generate_future_idxs(trajectories)

        elif self.generation_type == "random":
            idxs = self.generate_random_idxs(trajectories)

        elif self.generation_type == "final":
            idxs = self.generate_final_idx(trajectories)
        else:
            raise ValueError("Invalid generation type")
        return idxs

    def normalize_state(self, goal_state: np.ndarray, goal_feat_size: int = 2) -> np.ndarray:
        """
        Normalize and clip the state to be compatible with the agent.

        Args:
            state (np.ndarray): The state to be normalized and clipped.

        Returns:
            np.ndarray: The normalized and clipped state.
        """
        goal_state = np.clip(goal_state, self.observation_space.low[-goal_feat_size:], self.observation_space.high[-goal_feat_size:])
        goal_state = (goal_state - self.observation_space.low[-goal_feat_size:]) / (
            self.observation_space.high[-goal_feat_size:] - self.observation_space.low[-goal_feat_size:]
        )
        return goal_state

    def her_augmentation(self, sampled_td: TensorDictBase):
        if len(sampled_td.shape) == 1:
            sampled_td = sampled_td.unsqueeze(0)
        b, t = sampled_td.shape
        splitted_td = _split_and_pad_sequence(sampled_td, torch.tensor([t]))
        splitted_achieved_goals = splitted_td.get("achieved_state")

        # get indices for each trajectory
        idxs = self.generate_sample_idxs(torch.tensor([t]))

        # create new goals based idxs
        new_goals = []
        for i, ids in enumerate(idxs):
            new_goals.append(splitted_achieved_goals[i][ids])

        # calculate rewards given new desired goals and old achieved goals
        vmap_rewards = torch.vmap(self.reward_function)
        rewards = []
        for ach, des in zip(splitted_achieved_goals, new_goals):
            rewards.append(vmap_rewards(ach[: des.shape[0], :], des))

        cat_rewards = torch.cat(rewards).reshape(b, t, self.samples, -1)
        cat_rewards = cat_rewards.transpose(1, 2).flatten(0, 1).float()
        cat_new_goals = torch.cat(new_goals).reshape(b, t, self.samples, -1)
        cat_new_goals = cat_new_goals.transpose(1, 2).flatten(0, 1)
        achieved_state = splitted_achieved_goals.repeat_interleave(self.samples, dim=0)

        norm_new_goals = self.normalize_state(cat_new_goals, goal_feat_size=cat_new_goals.shape[-1])
        norm_achieved_state = self.normalize_state(achieved_state, goal_feat_size=achieved_state.shape[-1])

        augmentation_obs = torch.cat([norm_achieved_state, norm_new_goals], dim=-1).float()
        # TODO: normalize
        # repeat and then do cat(obs[obssize_withoutgoal:], new goals)

        obs_repeated = sampled_td.get("observation").repeat_interleave(
            self.samples, dim=0
        )
        goal_feat_size = cat_new_goals.shape[-1]
        obs_new_goals = torch.cat(
            [obs_repeated[:, :, :-goal_feat_size], norm_new_goals], dim=-1
        ).float()

        truncated_repeated = sampled_td.get("next")["truncated"].repeat_interleave(
            self.samples, dim=0
        )

        augmentation_td = TensorDict(
            {
                "observation": obs_new_goals,
                "action": sampled_td.get("action").repeat_interleave(
                    self.samples, dim=0
                ),
                # "achieved_state": achieved_state,
                # "desired_state": cat_new_goals,
                "reward": cat_rewards,
                "next": {
                    "observation": augmentation_obs,
                    "truncated": truncated_repeated,
                    "done": cat_rewards.bool(),
                    "reward": cat_rewards,
                },
            },
            batch_size=(b * self.samples, t),
        )
        return augmentation_td


def dense_reward_function(state, goal, goal_thresholds=20):
    errors = - torch.sum(torch.abs(state - goal), dim=-1)
    return errors

def sparse_reward_function(state, goal, goal_thresholds=20):
    errors = torch.abs(state - goal)
    return torch.where(errors <= goal_thresholds, 1, 0).prod(-1)
