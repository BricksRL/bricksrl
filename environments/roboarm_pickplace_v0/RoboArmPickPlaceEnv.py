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

    observation_ranges = {
        "GM": (-148, -45),  # Grab motor range is 0-180
        "HM": (-150, 10),
        "LM": (10, 100),
        "RM": (-179, 179),  # Rotation motor needs to be place in the center
    }
    observation_key = "observation"
    pixels_observation_key = "pixels"

    def __init__(
        self,
        max_episode_steps: int = 50,
        sleep_time: float = 0.0,
        verbose: bool = False,
        reward_signal: str = "dense",
        camera_id: int = 0,
        image_size: Tuple[int, int] = (64, 64),
        human_control: bool = False,
        use_vip_reward: bool = False,
        target_image_path: str = None,
        mixed_observation: bool = True,
    ):
        self.sleep_time = sleep_time

        assert reward_signal in [
            "dense",
            "sparse",
        ], "Reward signal must be dense or sparse."
        self.reward_signal = reward_signal
        self.max_episode_steps = max_episode_steps
        self.image_size = image_size
        self.human_control = human_control

        self.camera = cv2.VideoCapture(int(camera_id))
        self._batch_size = torch.Size([1])

        if target_image_path is not None:
            target_image = np.load(target_image_path)
        else:
            target_image = np.load(
                "environments/roboarm_pickplace_v0/pickplace_green100_target.npy"
            )
        self.target_image = target_image
        self.use_vip_reward = use_vip_reward
        self.mixed_observation = mixed_observation

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
                self.observation_ranges["GM"],
                self.observation_ranges["HM"],
                self.observation_ranges["LM"],
                self.observation_ranges["RM"],
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
        resized_frame = cv2.resize(frame, self.image_size)
        shape = resized_frame.shape
        pixels_observation_spec = BoundedTensorSpec(
            low=torch.zeros((1,) + shape, dtype=torch.int64),
            high=torch.ones((1,) + shape, dtype=torch.int64) * 255,
            dtype=torch.int64,
        )
        if self.mixed_observation:
            self.observation_spec = CompositeSpec(
                {
                    self.observation_key: observation_spec,
                    self.pixels_observation_key: pixels_observation_spec,
                },
                shape=(1,),
            )
        else:
            self.observation_spec = CompositeSpec(
                {
                    self.pixels_observation_key: pixels_observation_spec,
                },
                shape=(1,),
            )

        _ = self.init_camera_position()

        super().__init__(
            action_dim=self.action_dim, state_dim=self.state_dim, verbose=verbose
        )

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

    def get_reset_tensordict(self, **kwargs) -> TensorDictBase:
        """ """
        if self.use_vip_reward:
            return TensorDict(
                {
                    "goal_image": torch.from_numpy(self.target_image)
                    .to(torch.int64)
                    .unsqueeze(0),
                },
                batch_size=[
                    1,
                ],
            )
        return TensorDict({},batch_size=[1])

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        """
        Reset the environment and return the initial state.

        Returns:
            TensorDictBase: The initial state of the environment.
        """
        # TODO solve this fake action sending before to receive first state
        self.episode_step_iter = 0
        action = np.zeros(self.action_dim)
        self.send_to_hub(action)
        time.sleep(self.sleep_time)
        observation = self.read_from_hub()

        ret, frame = self.camera.read()
        resized_frame = cv2.resize(frame, self.image_size)

        return TensorDict(
            {
                self.observation_key: torch.tensor(observation).float(),
                self.pixels_observation_key: torch.from_numpy(resized_frame)[
                    None, :
                ].to(torch.int64),
                # "goal_image": torch.from_numpy(self.target_image)
                # .to(torch.int64)
                # .unsqueeze(0),
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
        resized_frame = cv2.resize(frame, self.image_size)
        next_tensordict = TensorDict(
            {
                self.observation_key: torch.tensor(next_observation).float(),
                self.pixels_observation_key: torch.from_numpy(resized_frame)[
                    None, :
                ].to(torch.int64),
                "reward": torch.tensor([reward]).float(),
                "done": torch.tensor([done]).bool(),
                # "goal_image": torch.from_numpy(self.target_image)
                # .to(torch.int64)
                # .unsqueeze(0),
            },
            batch_size=[1],
        )  # .to(tensordict.device)

        # increment episode step counter
        self.episode_step_iter += 1
        if self.episode_step_iter >= self.max_episode_steps:
            next_tensordict.set("done", torch.tensor([True]))
        return next_tensordict
