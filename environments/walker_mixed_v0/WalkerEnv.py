import time
from typing import Tuple

import cv2

import numpy as np

import torch

from environments.base.base_env import BaseEnv
from tensordict import TensorDict, TensorDictBase
from torchrl.data.tensor_specs import BoundedTensorSpec, CompositeSpec


class WalkerMixedEnv_v0(BaseEnv):
    """ """

    action_dim = 4  # (lf_value, lb_value, rf_value, rb_value)
    # angles are in range [-180, 179]
    state_dim = 7  # (lf_angle, rf_angle, lb_angle, rb_angle, pitch, roll, acc_x)
    max_acc = 3000
    motor_range = (-179, 179)
    pitch_roll_range = (-50, 50)
    vec_observation_key = "vec_observation"
    image_observation_key = "image_observation"
    original_image_key = "original_image"

    def __init__(
        self,
        max_episode_steps: int = 50,
        sleep_time: float = 0.0,
        verbose: bool = False,
        camera_id: int = 0,
        image_path: str = "environments/walker_mixed_v0/img/robodog.jpg",
        auto_obj_detection: bool = False,
        tracker_type: str = "KCF",
    ):
        self.sleep_time = sleep_time
        self.max_acc = 3000
        self._batch_size = torch.Size([1])
        self.max_episode_steps = max_episode_steps
        self.tracker_type = tracker_type
        self.auto_obj_detection = auto_obj_detection
        self.camera = cv2.VideoCapture(int(camera_id))
        # Initialize ORB detector
        self.orb = cv2.ORB_create()
        self.object_found = False
        self.tracker_initialized = False
        self.initial_center_y = None
        # Initialize BFMatcher to track features
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.keypoints, self.descriptors, self.ref_img = (
            self.load_image_and_compute_features(image_path)
        )
        if self.descriptors is None:
            raise ValueError("Failed to load reference image.")

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
            key (str): The key of the state to be normalized.

        Returns:
            torch.Tensor: The normalized state.
        """
        state = (torch.from_numpy(state) - self.observation_spec[key].space.low) / (
            self.observation_spec[key].space.high - self.observation_spec[key].space.low
        )
        return state

    def compute_features(self, image: np.array):
        # Function to compute keypoints and descriptors for a given image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        return keypoints, descriptors

    def load_image_and_compute_features(self, image_path: str = "./img/robodog.jpg"):
        # Load an image and compute its features
        img = cv2.imread(image_path)
        if img is not None:
            kp, des = self.compute_features(img)
            return kp, des, img
        else:
            print(f"Failed to load image: {image_path}")
            return None, None, None

    def init_tracker_drawing(
        self,
    ):

        while not self.tracker_initialized:
            ret, frame = self.camera.read()
            if not ret:
                raise Exception("Failed to read frame from video source.")
            print("\rPress 'y' to select the object to track.", end="", flush=True)
            cv2.imshow("Object Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("y"):
                # Function to initialize the tracker with a user-selected bounding box
                bbox = cv2.selectROI("Object Detection", frame, False)
                cv2.destroyAllWindows()
                self.tracker_initialized = True
                self.tracker.init(frame, bbox)
                self.initial_center_y = bbox[1] + bbox[3] / 2
            else:
                continue
        return frame

    def init_tracker_with_example_image(
        self,
    ):
        while not self.tracker_initialized:
            ret, frame = self.camera.read()
            if not ret:
                raise Exception("Failed to read frame from video source.")

            keypoints_frame, descriptors_frame = self.compute_features(frame)
            if descriptors_frame is not None:
                matches = self.bf.match(self.descriptors, descriptors_frame)
                matches = sorted(matches, key=lambda x: x.distance)

                if len(matches) > 10:
                    points = np.zeros((len(matches), 2), dtype=np.float32)
                    for i, match in enumerate(matches):
                        points[i, :] = keypoints_frame[match.trainIdx].pt

                    x, y, w, h = cv2.boundingRect(points.astype(np.float32))
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                cv2.imshow("Object Detection", frame)

                # Prompt user for input within the OpenCV window
                print(
                    "\rPress 'y' if the object is correct, 'n' to continue searching.",
                    end="",
                    flush=True,
                )
                # while not user_response:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("y"):
                    self.tracker_initialized = True
                    self.tracker.init(frame, (x, y, w, h))
                    self.initial_center_y = y + h / 2
                elif key == ord("n"):
                    self.tracker_initialized = False
                    break  # Move to the next frame without initializing the tracker
                else:
                    self.tracker_initialized = False
                    continue
            else:
                cv2.imshow("Object Detection", frame)
        return frame

    def init_tracker(self):
        if self.auto_obj_detection:
            frame = self.init_tracker_with_example_image()
        else:
            frame = self.init_tracker_drawing()
        return frame

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        """
        Reset the environment and return the initial state.

        Returns:
            TensorDictBase: The initial state of the environment.
        """
        # TODO solve this fake action sending before to receive first state
        self.episode_step_iter = 0
        self.current_upward_distance = 0
        self.tracker_initialized = False
        self.walking_path = []
        if tensordict is not None:
            action = tensordict.get("action").numpy().squeeze()
        else:
            action = np.zeros(self.action_dim)
        self.send_to_hub(action)
        time.sleep(self.sleep_time)
        observation = self.read_from_hub()
        norm_obs = self.normalize_state(observation, self.vec_observation_key)
        self.tracker = get_tracker(self.tracker_type)
        frame = self.init_tracker()
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

        return reward.item(), done

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """ """
        # Send action to hub to receive next state
        action = tensordict.get("action").numpy().squeeze()
        self.send_to_hub(action)
        time.sleep(
            self.sleep_time
        )  # we need to wait some time for sensors to read and to
        # receive the next state
        next_observation = self.read_from_hub()
        done = False
        # receive the next state
        ret, frame = self.camera.read()
        success, bbox = self.tracker.update(frame)
        if success:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

            center = (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))
            self.walking_path.append(center)
            upward_distance = self.initial_center_y - center[1]
            # TODO: could be also just +1 if positive and -1 if negative
            reward = upward_distance - self.current_upward_distance
            self.current_upward_distance = upward_distance
            for i in range(1, len(self.walking_path)):
                if self.walking_path[i - 1] is None or self.walking_path[i] is None:
                    continue
                cv2.line(
                    frame,
                    self.walking_path[i - 1],
                    self.walking_path[i],
                    (0, 255, 0),
                    3,
                )

            # Display the total upward distance traveled on the frame
            distance_text = f"Upward Distance: {upward_distance}px"
            cv2.putText(
                frame,
                distance_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                2,
            )
            cv2.imshow("Tracking", frame)
            cv2.waitKey(1)

        else:
            reward = 0  # No reward if tracking fails
            done = True  # End episode if tracking fails

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


def angular_difference(angle1, angle2):
    # Calculate the difference in angles, wrapped between -180 and 180
    difference = (angle2 - angle1 + 180) % 360 - 180
    return abs(difference)  # Return the absolute value of the difference


def get_tracker(type="KCF"):
    if type == "KCF":
        return cv2.TrackerKCF_create()
    elif type == "BOOSTING":
        return cv2.legacy.TrackerBoosting_create()
    elif type == "MIL":
        return cv2.legacy.TrackerMIL_create()
    elif type == "TLD":
        return cv2.legacy.TrackerTLD_create()
    elif type == "MEDIANFLOW":
        return cv2.legacy.TrackerMedianFlow_create()
    elif type == "MOSSE":
        return cv2.legacy.TrackerMOSSE_create()
    elif type == "CSRT":
        return cv2.legacy.TrackerCSRT_create()
    else:
        print(
            "Invalid tracker type! Please use one of the following: KCF, BOOSTING, MIL, TLD, MEDIANFLOW, MOSSE, CSRT"
        )
