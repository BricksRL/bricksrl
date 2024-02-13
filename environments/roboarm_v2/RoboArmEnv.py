import time
from typing import Tuple

import gym
import numpy as np
import cv2

from environments.base.base_env import BaseEnv


class RoboArmEnv_v2(BaseEnv):
    """ """

    def __init__(
        self,
        max_episode_steps: int = 50,
        sleep_time: float = 0.0,
        verbose: bool = False,
        reward_signal: str = "dense",
        camera_idx: int = 0,
        img_dim: Tuple[int, int] = (64, 64),
        color_range_low: Tuple[int, int, int] = (46, 141, 116),
        color_range_high: Tuple[int, int, int] = (179, 240,255),
    ):
        action_dim = 2  # (Grab_motor_action, high_motor_action, low_motor_action, rotation_motor_action)
        # angles are in range [-180, 179]

        state_dim = 2  # (GM, HM, LM, RM, GGM, GHM, GLM, GRM)
        self.sleep_time = sleep_time
        self.camera = cv2.VideoCapture(camera_idx)
        # TODO check if camera is opened
        assert reward_signal in ["dense", "sparse"], "Reward signal must be dense or sparse."
        # TODO: add dense reward signal
        self.reward_signal = reward_signal
        self.max_episode_steps = max_episode_steps

        self.action_space = gym.spaces.Box(
            low=-np.ones(action_dim), high=np.ones(action_dim), shape=(action_dim,)
        )
        # grab motor range
        # (-230, -148) left side closed (-148, -44) -> starting -44

        # high motor range
        # (-60, 140) -> starting 123

        # low motor range
        # (-190, -10) -> starting -19

        # rotation motor range
        # (-360, 360) -> starting 45
        motor_ranges = {
            #"GM": (-230, -148),
            "HM": (-60, 140),
            #"LM": (-190, -10),
            "RM": (-900, 900),
        }
        self.motor_ranges = motor_ranges

        # Observation 4 motors (GM, HM, LM, RM)
        # Extend the observation space to include pixel/image observations
        image_obs_dim = (img_dim[0], img_dim[1], 3)  # Example dimensions for RGB image
        self.observation_space = gym.spaces.Dict({
            'state': gym.spaces.Box(
                low=np.array([
                    #self.motor_ranges["GM"][0],
                    self.motor_ranges["HM"][0],
                    #self.motor_ranges["LM"][0],
                    self.motor_ranges["RM"][0],
                ]),
                high=np.array([
                    #self.motor_ranges["GM"][1],
                    self.motor_ranges["HM"][1],
                    #self.motor_ranges["LM"][1],
                    self.motor_ranges["RM"][1],

                ])
            ),
            'image': gym.spaces.Box(
                low=0,
                high=255,
                shape=image_obs_dim,
                dtype=np.uint8
            )
        })

        # Define the radius and color of the circle (BGR format for color)
        self.img_dim = img_dim
        self.radius = 50
        self.color = (0, 0, 255)  # Red color
        self.color_range_low = color_range_low
        self.color_range_high = color_range_high

        super().__init__(action_dim=action_dim, state_dim=state_dim, verbose=verbose)

    def sample_random_action(self) -> np.ndarray:
        """
        Sample a random action from the action space.

        Returns:
            np.ndarray: A random action from the action space.
        """
        action = np.random.uniform(
            self.action_space.minimum, self.action_space.maximum, size=self.action_dim
        )
        return action

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """
        Normalize and clip the state to be compatible with the agent.

        Args:
            state (np.ndarray): The state to be normalized and clipped.

        Returns:
            np.ndarray: The normalized and clipped state.
        """
        state = np.clip(state, self.observation_space.low, self.observation_space.high)
        state = (state - self.observation_space.low) / (
            self.observation_space.high - self.observation_space.low
        )
        return state

    def reset(self) -> np.ndarray:
        """
        Reset the environment and return the initial state.

        Returns:
            np.ndarray: The initial state of the environment.
        """
        # TODO solve this fake action sending before to receive first state
        self.episode_step_iter = 0
        action = np.zeros(self.action_dim) # to bring robot in starting position!
        self.send_to_hub(action)
        time.sleep(self.sleep_time)
        self.observation = self.read_from_hub()
        _, frame = self.camera.read()

        # sample random goal state
        height, width, _ = frame.shape
        
        # Generate random center coordinates for the circle
        self.goal_center_x = np.random.randint(0, width)
        self.goal_center_y = np.random.randint(0, height)

        # Draw the circle on the frame
        cv2.circle(frame, (self.goal_center_x, self.goal_center_y), self.radius, self.color, -1)  # -1 fill the circle
        frame = cv2.resize(frame, self.img_dim)
        if self.verbose:
            print("Raw state received: ", self.observation)
            print("Goal state: ", self.goal_state)
    
        return {"state": self.normalize_state(self.observation.squeeze()), "image": frame}

    def get_contours(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
        # Find contours in the green mask
        contours, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    @staticmethod
    def is_overlapping(x1, y1, r1, x2, y2, r2):
        # Calculate the distance between the two centers
        distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        return distance < (r1 + r2)

    def reward(
        self,
        frame: np.ndarray
    ) -> Tuple[float, bool]:
        """Reward function of walker.

        Goal: Increase forward velocity, estimated from acceleration.

        Args:
            state (np.ndarray): The current state.
            action (np.ndarray): The action taken.
            next_state (np.ndarray): The next state.
            delta_t (float): The time step duration.

        Returns:
            Tuple[float, bool]: The reward received and a boolean indicating whether the episode is done.
        """

        done = False
        reward = 0
        contours = self.get_contours(frame)
        for contour in contours:
            # Optional: Draw green contours for visualization
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)

            # Simple overlap detection by checking the bounding rect or use more complex logic
            x, y, w, h = cv2.boundingRect(contour)
            if self.is_overlapping(self.goal_center_x, self.goal_center_y, self.radius, x + w/2, y + h/2, max(w, h)/2):
                done = True
                reward = 1
                break
        
        return reward, done
    

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Perform the given action and return the next state, reward, done status, and truncation status.

        Args:
            action (np.ndarray): The action to perform.

        Returns:
            Tuple[np.ndarray, float, bool, bool, dict]: A tuple containing the next state, the reward
            received for performing the action, a boolean indicating whether the episode is done,
            a boolean indicating whether the episode is truncated, and an empty dictionary.
        """
        truncated = False
        # Send action to hub to receive next state
        self.send_to_hub(action)
        time.sleep(
            self.sleep_time
        )  # we need to wait some time for sensors to read and to

        # receive the next state
        next_observation = self.read_from_hub()
        _, frame = self.camera.read()

        # sample random goal state
        height, width, _ = frame.shape
        
        # Generate random center coordinates for the circle
        self.goal_center_x = np.random.randint(0, width)
        self.goal_center_y = np.random.randint(0, height)

        # Draw the circle on the frame
        cv2.circle(frame, (self.goal_center_x, self.goal_center_y), self.radius, self.color, -1)  # -1 fill the circle
        
        current_time = time.time()
        # calc reward and done
        reward, done = self.reward(
            frame
        )
        if self.verbose:
            print("State", self.observation)
            print("Action", action)
            print("Next state", next_observation)
            print("Reward", reward)
            print("Goal state", self.goal_state)
        # set next state as current state
        self.observation = next_observation
        if done:
            cv2.putText(frame, "Solved", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        frame = cv2.resize(frame, self.img_dim)
        # increment episode step counter
        self.episode_step_iter += 1
        if self.episode_step_iter >= self.max_episode_steps:
            truncated = True
        self.dt = current_time
        return (
            {"state": self.normalize_state(self.observation.squeeze()), "image": frame},
            reward,
            done,
            truncated,
            {},
        )
