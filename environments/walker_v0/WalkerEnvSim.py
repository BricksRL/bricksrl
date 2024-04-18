from typing import Tuple

import numpy as np

import torch

from environments.base.base_env import BaseSimEnv
from tensordict import TensorDict, TensorDictBase
from torchrl.data.tensor_specs import BoundedTensorSpec, CompositeSpec


class WalkerEnvSim_v0(BaseSimEnv):
    """ """

    action_dim = 4  # (lf_value, lb_value, rf_value, rb_value)
    # angles are in range [-180, 179]
    state_dim = 7  # (lf_angle, rf_angle, lb_angle, rb_angle, pitch, roll, acc_x)

    motor_range = (-179, 179)
    pitch_roll_range = (-50, 50)
    observation_key = "vec_observation"
    original_vec_observation_key = "original_vec_observation"

    def __init__(
        self,
        max_episode_steps: int = 50,
        noise: float = 0.1,
        low_action_angle: int = -100,
        high_action_angle: int = 0,
        verbose: bool = False,
    ):
        self.max_acc = 3000
        self._batch_size = torch.Size([1])
        self.max_episode_steps = max_episode_steps
        self.noise = noise
        self.low_action_angle = low_action_angle
        self.high_action_angle = high_action_angle
        self.current_leg_angles = None

        self.action_spec = BoundedTensorSpec(
            low=-torch.ones((1, self.action_dim)),
            high=torch.ones((1, self.action_dim)),
            shape=(1, self.action_dim),
        )

        max_acc_range = (-self.max_acc, self.max_acc)
        observation_spec = BoundedTensorSpec(
            low=torch.tensor(
                [
                    [
                        self.motor_range[0],
                        self.motor_range[0],
                        self.motor_range[0],
                        self.motor_range[0],
                        self.pitch_roll_range[0],
                        self.pitch_roll_range[0],
                        max_acc_range[0],
                    ]
                ]
            ),
            high=torch.tensor(
                [
                    [
                        self.motor_range[1],
                        self.motor_range[1],
                        self.motor_range[1],
                        self.motor_range[1],
                        self.pitch_roll_range[1],
                        self.pitch_roll_range[1],
                        max_acc_range[1],
                    ]
                ]
            ),
        )

        self.observation_spec = CompositeSpec(shape=(1,))
        self.observation_spec.set(self.observation_key, observation_spec)
        super().__init__(
            action_dim=self.action_dim, state_dim=self.state_dim, verbose=verbose
        )

    def normalize_state(self, state: np.ndarray) -> torch.Tensor:
        """
        Normalize the state to be processed by the agent.

        Args:
            state (np.ndarray): The state to be normalized.

        Returns:
            torch.Tensor: The normalized state.
        """
        state = (
            torch.from_numpy(state)
            - self.observation_spec[self.observation_key].space.low
        ) / (
            self.observation_spec[self.observation_key].space.high
            - self.observation_spec[self.observation_key].space.low
        )
        return state

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        """
        Reset the environment and return the initial state.

        Returns:
            TensorDictBase: The initial state of the environment.
        """
        # TODO solve this fake action sending before to receive first state
        self.episode_step_iter = 0

        observation = self.observation_spec[self.observation_key].rand().numpy()
        self.current_leg_angles = observation[0, :4]
        norm_observation = self.normalize_state(observation)
        return TensorDict(
            {
                self.observation_key: norm_observation.float(),
                self.original_vec_observation_key: torch.from_numpy(
                    observation
                ).float(),
            },
            batch_size=[1],
        )

    def reward(
        self,
        action: np.ndarray,
        next_state: np.ndarray,
    ) -> Tuple[float, bool]:
        """Reward function of walker.

        Goal: Increase forward velocity, estimated from acceleration.

        Args:
            action (np.ndarray): The action taken.
            next_state (np.ndarray): The next state.

        Returns:
            Tuple[float, bool]: The reward received and a boolean indicating whether the episode is done.
        """

        done = False
        # pitch and roll need to stay in range [-75, 75] outside done = True
        pitch, roll = next_state[:, -3], next_state[:, -2]
        if np.abs(pitch) > 100 or np.abs(roll) > 100:
            done = True
            reward = 0
            return reward, done

        (
            lf_angle,
            rf_angle,
            lb_angle,
            rb_angle,
            pitch,
            roll,
            acc_x,
        ) = next_state.squeeze()

        # we want actions to be negative and high
        # action is in range [-1, 1] over 4 dims -> sum is in range [-4, 4] -> divide by 4 to get in range [-1, 1]
        action_reward = -np.sum(action) / 4 / 10
        # Take this off we dont want them to be similar otherwise we cant adapt for noise in the system
        # actions should ideally be similar something like [-0.75, -0.75, -0.75, -0.75]
        # action_std_reward = -np.std(action)

        # we want lf_angle and rb_angle to be synchronized and rf_angle and lb_angle to be synchronized
        # divide by 180 to get in range [-1, 0]
        lf_rb_diff_reward = -angular_difference(lf_angle, rb_angle) / 180
        rf_lb_diff_reward = -angular_difference(rf_angle, lb_angle) / 180

        # we want lf_rb and rf_lb to be 180° apart
        # divide by 180 to get in range [-1, 0]
        lf_rf_180_reward = -(180 - angular_difference(lf_angle, rf_angle)) / 180
        lb_rb_180_reward = -(180 - angular_difference(lb_angle, rb_angle)) / 180

        if self.verbose:
            # TODO: maybe we want add those values as an info dict to the env step return
            print("action_reward", action_reward)
            # print("action_std_reward", action_std_reward)
            print("lf_rb_diff_reward", lf_rb_diff_reward)
            print("rf_lb_diff_reward", rf_lb_diff_reward)
            print("lf_rf_180_reward", lf_rf_180_reward)

        reward = (
            action_reward
            # + action_std_reward
            + lf_rb_diff_reward
            + rf_lb_diff_reward
            + lf_rf_180_reward
            + lb_rb_180_reward
        )

        return reward.item(), done

    @staticmethod
    def transform_range(value, old_min, old_max, new_min, new_max):
        """
        Transform a value from one range to another.

        Parameters:
        value (float): The value to transform.
        old_min (float): The minimum value of the old range.
        old_max (float): The maximum value of the old range.
        new_min (float): The minimum value of the new range.
        new_max (float): The maximum value of the new range.

        Returns:
        float: The transformed value.
        """
        # Compute the scale factor between the old and new ranges
        scale = (new_max - new_min) / (old_max - old_min)
        # Apply the transformation
        return new_min + (value - old_min) * scale

    @staticmethod
    def normalize_angle(angle, low_angle=-180, high_angle=179, original_one_round=360):
        # Normalize angle to be within -179 to 179 degrees
        while angle <= low_angle:
            angle += original_one_round
        while angle > high_angle:
            angle -= original_one_round
        return angle

    def apply_action(self, action: np.ndarray) -> np.ndarray:

        noise = np.random.normal(0, self.noise, size=4)
        action += noise

        lf_value, lb_value, rf_value, rb_value = action
        # transform action range for motors
        lf_action = self.transform_range(
            lf_value, -1, 1, self.low_action_angle, self.high_action_angle
        )
        lb_action = self.transform_range(
            lb_value, -1, 1, self.low_action_angle, self.high_action_angle
        )
        rf_actopm = self.transform_range(
            rf_value, -1, 1, self.low_action_angle, self.high_action_angle
        )
        rb_action = self.transform_range(
            rb_value, -1, 1, self.low_action_angle, self.high_action_angle
        )

        (
            lf_angle,
            rf_angle,
            lb_angle,
            rb_angle,
        ) = self.current_leg_angles.squeeze()

        new_lf_angle = self.normalize_angle(lf_angle + lf_action)
        new_lb_angle = self.normalize_angle(lb_angle + lb_action)
        new_rf_angle = self.normalize_angle(rf_angle + rf_actopm)
        new_rb_angle = self.normalize_angle(rb_angle + rb_action)

        self.current_leg_angles = np.array(
            [
                [
                    new_lf_angle,
                    new_rf_angle,
                    new_lb_angle,
                    new_rb_angle,
                ]
            ]
        )
        return self.current_leg_angles

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """ """
        # Send action to hub to receive next state
        action = tensordict.get("action").cpu().numpy().squeeze()

        # receive the next state
        next_observation = self.apply_action(action)

        # add zeros for pitch, roll and acc_x
        next_observation = np.concatenate((next_observation, np.zeros((1, 3))), axis=1)

        # calc reward and done
        reward, done = self.reward(
            action=action,
            next_state=next_observation,
        )
        next_tensordict = TensorDict(
            {
                self.observation_key: self.normalize_state(next_observation).float(),
                self.original_vec_observation_key: torch.from_numpy(
                    next_observation
                ).float(),
                "reward": torch.tensor([reward]).float(),
                "done": torch.tensor([done]).bool(),
            },
            batch_size=[1],
        )

        # increment episode step counter
        self.episode_step_iter += 1
        if self.episode_step_iter >= self.max_episode_steps:
            next_tensordict.set("done", torch.tensor([True]))
        return next_tensordict


def angular_difference(angle1, angle2):
    # Calculate the difference in angles, wrapped between -180 and 180
    difference = (angle2 - angle1 + 180) % 360 - 180
    return abs(difference)  # Return the absolute value of the difference