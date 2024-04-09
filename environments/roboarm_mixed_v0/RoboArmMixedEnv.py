import random
import time
from typing import Tuple

import cv2

import numpy as np
import torch

from environments.base.base_env import BaseEnv
from tensordict import TensorDict, TensorDictBase
from torchrl.data.tensor_specs import BoundedTensorSpec, CompositeSpec


def random_center_position(frame):
    height, width = frame.shape[:2]
    center_x = random.randint(0, width)
    center_y = random.randint(0, height)
    return center_x, center_y


class RoboArmMixedEnv_v0(BaseEnv):
    """ """

    action_dim = 3  # (high_motor_action, low_motor_action, rotation_motor_action)
    # angles are in range [-180, 179]
    state_dim = 3  # (HM, LM, RM)

    motor_ranges = {
        "HM": (-150, 10),
        "LM": (10, 70),
        "RM": (-180, 179),
    }
    goal_color = (0, 255, 0)  # red
    vec_observation_key = "vec_observation"
    image_observation_key = "image_observation"
    original_image_key = "original_image"

    def __init__(
        self,
        max_episode_steps: int = 50,
        sleep_time: float = 0.0,
        verbose: bool = False,
        reward_signal: str = "dense",
        camera_id: int = 0,
        goal_radius: float = 50,
    ):
        self.sleep_time = sleep_time

        assert reward_signal in [
            "dense",
            "sparse",
        ], "Reward signal must be dense or sparse."
        self.reward_signal = reward_signal
        self.max_episode_steps = max_episode_steps
        self.goal_radius = goal_radius
        self.prev_reward = 0.0  # if we dont detect contours we can still give a reward
        self.contour_threshold = 10
        self.distance_threshold = 40
        self.camera = cv2.VideoCapture(int(camera_id))
        self._batch_size = torch.Size([1])

        self.action_spec = BoundedTensorSpec(
            low=-torch.ones((1, self.action_dim)),
            high=torch.ones((1, self.action_dim)),
            shape=(1, self.action_dim),
        )

        # Observation 3 motors (HM, LM, RM)
        observation_spec = BoundedTensorSpec(
            low=torch.tensor(
                [
                    [
                        self.motor_ranges["HM"][0],
                        self.motor_ranges["LM"][0],
                        self.motor_ranges["RM"][0],
                    ]
                ]
            ),
            high=torch.tensor(
                [
                    [
                        self.motor_ranges["HM"][1],
                        self.motor_ranges["LM"][1],
                        self.motor_ranges["RM"][1],
                    ]
                ]
            ),
        )
        # get initial observation to define image observation spec
        ret, frame = self.camera.read()
        if not ret:
            raise ValueError("Camera not available.")
        frame = cv2.resize(frame, (64, 64))
        shape = frame.shape
        # img_dtype = frame.dtype
        image_observation_spec = BoundedTensorSpec(
            low=torch.zeros((1,) + shape, dtype=torch.uint8),
            high=torch.ones((1,) + shape, dtype=torch.uint8) * 255,
        )

        self.observation_spec = CompositeSpec(shape=(1,))
        self.observation_spec.set(self.vec_observation_key, observation_spec)
        self.observation_spec.set(self.image_observation_key, image_observation_spec)
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

    def _get_contours(self, frame):
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define range of green color in HSV to define those values and tune them use: https://github.com/BY571/python_webcam/blob/main/color_track_bar.py
        lower_green = (135, 155, 86)
        upper_green = (179, 255, 255)
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        # Find contours in the green mask
        contours, _ = cv2.findContours(
            green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        return contours

    def _draw_contours(self, frame, contours):
        for contour in contours:
            # Optional: Draw green contours for visualization
            cv2.drawContours(frame, [contour], -1, (0, 0, 255), 2)

    def _draw_goal_circle(self, frame):
        # Draw the circle on the frame
        cv2.circle(
            frame,
            (self.goal_center_x, self.goal_center_y),
            self.goal_radius,
            self.goal_color,
            -1,
        )  # Red color

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
        norm_obs = self.normalize_state(observation, self.vec_observation_key)

        ret, frame = self.camera.read()
        # get random goal location
        self.goal_center_x, self.goal_center_y = random_center_position(frame)
        self._draw_goal_circle(frame)
        self._draw_contours(frame, self._get_contours(frame))
        resized_frame = cv2.resize(frame, (64, 64))

        return TensorDict(
            {
                self.vec_observation_key: norm_obs.float(),
                self.image_observation_key: torch.from_numpy(resized_frame)[
                    None, :
                ].float(),
                self.original_image_key: torch.from_numpy(frame)[None, :].to(
                    torch.uint8
                ),
            },
            batch_size=[1],
        )

    def calculate_distance(self, contour):
        x, y, w, h = cv2.boundingRect(contour)
        center_x, center_y = x + w // 2, y + h // 2
        distance = int(
            (
                (center_x - self.goal_center_x) ** 2
                + (center_y - self.goal_center_y) ** 2
            )
            ** 0.5
        )
        return distance, center_x, center_y

    def reward(
        self,
        frame: np.ndarray,
        contours: list,
    ) -> Tuple[float, bool]:
        """ """
        significant_contour_found = False
        done = False
        reward = 0.0

        distances = []
        for contour in contours:
            if (
                cv2.contourArea(contour) > self.contour_threshold
            ):  # Only consider contours with area > 10 as significant
                significant_contour_found = True
                cv2.drawContours(
                    frame, [contour], -1, (0, 0, 255), 2
                )  # Draw each contour
                distance, _, _ = self.calculate_distance(contour)
                distances.append(distance)
                if distance <= self.distance_threshold:
                    done = True
                    break
        cv2.putText(
            frame,
            f"Distance: {np.mean(distances)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
        )
        if self.reward_signal == "dense":
            if significant_contour_found:
                reward = -np.mean(distances) / 100  # maybe min?
            else:
                reward = self.prev_reward
        elif self.reward_signal == "sparse":
            if done:
                reward = 1.0
        else:
            raise ValueError("Reward signal must be dense or sparse.")
        if done:
            cv2.putText(
                frame, "Done!", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
            )
        self.prev_reward = reward
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
        self._draw_goal_circle(frame)
        contours = self._get_contours(frame)

        # calc reward and done
        reward, done = self.reward(
            frame,
            contours,
        )
        # self._draw_contours(frame, contours=contours)
        resized_frame = cv2.resize(frame, (64, 64))
        next_tensordict = TensorDict(
            {
                self.vec_observation_key: self.normalize_state(
                    next_observation, self.vec_observation_key
                ).float(),
                self.image_observation_key: torch.from_numpy(resized_frame)[
                    None, :
                ].float(),
                self.original_image_key: torch.from_numpy(frame)[None, :].to(
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
