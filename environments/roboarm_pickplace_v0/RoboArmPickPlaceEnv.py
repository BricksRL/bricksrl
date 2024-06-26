import random
import time
from typing import Tuple

import cv2

import numpy as np
import torch

from environments.base.base_env import BaseEnv
from tensordict import TensorDict, TensorDictBase
from torchrl.data.tensor_specs import BoundedTensorSpec, CompositeSpec


class RoboArmPickPlaceEnv_v0(BaseEnv):
    """ """

    action_dim = (
        4  # (grab_motor, high_motor_action, low_motor_action, rotation_motor_action)
    )

    state_dim = 4  # (GM, HM, LM, RM)

    motor_ranges = {
        "GM": (-148, -45),  # Grab motor range is 0-180
        "HM": (-150, 10),
        "LM": (10, 100),
        "RM": (-179, 179),  # Rotation motor needs to be place in the center
    }
    vec_observation_key = "vec_observation"
    image_observation_key = "image_observation"
    # original_image_key = "original_image"

    def __init__(
        self,
        max_episode_steps: int = 50,
        sleep_time: float = 0.0,
        verbose: bool = False,
        reward_signal: str = "dense",
        camera_id: int = 0,
    ):
        self.sleep_time = sleep_time

        assert reward_signal in [
            "dense",
            "sparse",
        ], "Reward signal must be dense or sparse."
        self.reward_signal = reward_signal
        self.max_episode_steps = max_episode_steps

        self.camera = cv2.VideoCapture(int(camera_id))
        self._batch_size = torch.Size([1])

        # Define action spec
        self.action_spec = BoundedTensorSpec(
            low=-1,
            high=1,
            shape=(1, self.action_dim),
        )

        # Observation 3 motors (HM, LM, RM)
        # Define observation spec
        bounds = torch.tensor(
            [
                self.motor_ranges["GM"],
                self.motor_ranges["HM"],
                self.motor_ranges["LM"],
                self.motor_ranges["RM"],
            ]
        )

        low_bounds = bounds[:, 0].unsqueeze(0)
        high_bounds = bounds[:, 1].unsqueeze(0)

        observation_spec = BoundedTensorSpec(
            low=low_bounds,
            high=high_bounds,
            dtype=torch.float32,
        )
        # get initial observation to define image observation spec
        ret, frame = self.camera.read()
        if not ret:
            raise ValueError("Camera not available.")
        resized_frame = cv2.resize(frame, (64, 64))
        shape = resized_frame.shape
        image_observation_spec = BoundedTensorSpec(
            low=torch.zeros((1,) + shape, dtype=torch.uint8),
            high=torch.ones((1,) + shape, dtype=torch.uint8) * 255,
            dtype=torch.uint8,
        )

        self.observation_spec = CompositeSpec(
            {
                self.vec_observation_key: observation_spec,
                self.image_observation_key: image_observation_spec,
            },
            shape=(1,),
        )

        self.goal_positions = self.init_camera_position()

        super().__init__(
            action_dim=self.action_dim, state_dim=self.state_dim, verbose=verbose
        )

    def normalize_state(self, state: np.ndarray, key: str) -> torch.Tensor:
        """
        Normalize the state to be processed by the agent.

        Args:
            state (np.ndarray): The state to be normalized.

        Returns:
            torch.Tensor: The normalized state.
        """
        state = (torch.from_numpy(state) - self.observation_spec[key].space.low) / (
            self.observation_spec[key].space.high - self.observation_spec[key].space.low
        )
        return state

    def init_camera_position(
        self,
    ):
        print(
            "\nInitializing camera position... \nMake sure the robot is in the center of the frame.\nPlease press 'c' to continue..."
        )
        while True:
            ret, frame = self.camera.read()
            if not ret:
                print("Error: Can't receive frame. Exiting ...")
                break

            cv2.imshow("Init RobotPosition", frame)

            if cv2.waitKey(1) == ord("c"):
                break

        return

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        """
        Reset the environment and return the initial state.

        Returns:
            TensorDictBase: The initial state of the environment.
        """
        # TODO solve this fake action sending before to receive first state
        self.episode_step_iter = 0
        if tensordict is not None:
            action = tensordict.get("action").cpu().numpy().squeeze()
        else:
            action = np.zeros(self.action_dim)
        self.send_to_hub(action)
        time.sleep(self.sleep_time)
        observation = self.read_from_hub()
        # assert
        norm_obs = self.normalize_state(observation, self.vec_observation_key)

        ret, frame = self.camera.read()
        resized_frame = cv2.resize(frame, (64, 64))

        return TensorDict(
            {
                self.vec_observation_key: norm_obs.float(),
                self.image_observation_key: torch.from_numpy(resized_frame)[None, :].to(
                    torch.uint8
                ),
            },
            batch_size=[1],
        )

    def reward(
        self,
        frame: np.ndarray,
    ) -> Tuple[float, bool]:
        """ """
        # TODO: Find a way to classify if cup is in the goal location
        done = False
        reward = 0.0
        return reward, done

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """ """
        # Send action to hub to receive next state
        self.send_to_hub(tensordict.get("action").cpu().numpy().squeeze())
        time.sleep(
            self.sleep_time
        )  # we need to wait some time for sensors to read and to

        # receive the next state
        next_observation = self.read_from_hub()

        # get next frame
        ret, frame = self.camera.read()

        cv2.imshow("Camera", frame)
        cv2.waitKey(1)
        # calc reward and done
        reward, done = self.reward(
            frame,
        )
        resized_frame = cv2.resize(frame, (64, 64))
        next_tensordict = TensorDict(
            {
                self.vec_observation_key: self.normalize_state(
                    next_observation, self.vec_observation_key
                ).float(),
                self.image_observation_key: torch.from_numpy(resized_frame)[None, :].to(
                    torch.uint8
                ),
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
