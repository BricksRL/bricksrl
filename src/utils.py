from typing import List

import numpy as np
import tensordict as td
import torch
from environments import ALL_2WHEELER_ENVS, ALL_ROBOARM_ENVS, ALL_WALKER_ENVS
from moviepy.editor import concatenate_videoclips, ImageClip
from omegaconf import DictConfig
from tensordict import TensorDictBase
from tqdm import tqdm


def setup_check(robot: str, config: DictConfig):
    if robot == "2wheeler":
        assert (
            config.env.name in ALL_2WHEELER_ENVS
        ), f"You are trying to run a 2wheeler experiment but are using the env {config.env.name}, select one of {ALL_2WHEELER_ENVS}"
    elif robot == "walker":
        assert (
            config.env.name in ALL_WALKER_ENVS
        ), f"You are trying to run a walker experiment but are using the env {config.env.name}, select one of {ALL_WALKER_ENVS}"
    elif robot == "roboarm":
        assert (
            config.env.name in ALL_ROBOARM_ENVS
        ), f"You are trying to run a roboarm experiment but are using the env {config.env.name}, select one of {ALL_ROBOARM_ENVS}"


def data2numpy(data: list):
    """Convert a list of bytes to a numpy array."""
    return np.array(data)[None, :]


def handle_disconnect(_):
    print("Hub was disconnected.")


def tensordict2dict(td: TensorDictBase) -> dict:
    """Convert a TensorDict to a dictionary."""
    return {k: v.item() for k, v in td.items()}


def create_transition_td(
    observation: np.array,
    action: np.array,
    reward: np.array,
    next_observation: np.array,
    done: np.array,
    batch_size: int = 1,
):
    """Create a TensorDict from a transition tuple."""
    obs_t = torch.from_numpy(observation).float()[None, :]
    action_t = torch.from_numpy(action).float()[None, :]
    reward_t = torch.from_numpy(reward).float()[None, :]
    next_obs_t = torch.from_numpy(next_observation).float()[None, :]
    done_t = torch.from_numpy(done).bool()[None, :]

    return td.TensorDict(
        {
            "observation": obs_t,
            "action": action_t,
            "reward": reward_t,
            "next": {"observation": next_obs_t, "reward": reward_t, "done": done_t},
        },
        batch_size=batch_size,
    )


def logout(agent):
    # TODO save model or training data
    x = input("Do you want to save the model? (y/n)")
    save_dict = {}
    if x == "y":
        agent_dict = agent.get_agent_statedict()
        save_dict.update(agent_dict)
    x = input("Do you want to save the replay buffer? (y/n)")
    if x == "y":
        buffer_dict = agent.replay_buffer.state_dict()
        save_dict.update(buffer_dict)
    if len(save_dict) > 0:
        save_name = input("Enter the name of the file to save: ")
        torch.save(save_dict, save_name + ".pth")


def login(agent):
    x = input("Do you want to load the model? (y/n)")
    if x == "y":
        save_name = input("Enter the name of the file to load: ")
        agent.load_model(save_name)
    else:
        print("Model not loaded!")
    x = input("Do you want to load the replay buffer? (y/n)")
    if x == "y":
        save_name = input("Enter the name of the file to load: ")
        agent.load_replaybuffer(save_name)
    else:
        print("Buffer not loaded!")


def prefill_buffer(env, agent, num_episodes=10, stop_on_done=False):
    """
    Prefills the agent's replay buffer with experiences by running the environment for a specified number of episodes.

    Args:
    - env: gym.Env object representing the environment
    - agent: Agent object with an add_experience method to add experiences to the replay buffer
    - num_episodes: int, number of episodes to run the environment for

    Returns: None
    """
    if agent.name in ["sac", "td3"]:
        inpt = input("Press Enter to start prefilling episode: ")
        for e in tqdm(range(num_episodes), desc="Prefilling buffer"):
            print("Prefill episode: ", e)

            td = env.reset()
            done = False
            truncated = False
            while not done and not truncated:
                td = env.sample_random_action(td)
                td = env.step(td)
                agent.add_experience(td)
                done = td.get(("next", "done"))

                if done and stop_on_done:
                    inpt = input(
                        "Please reset the robot to the starting position and press Enter to continue or q to quit:"
                    )
                    if inpt == "q":
                        break
        print("Prefill done! Buffer size: ", agent.replay_buffer.__len__())


def convert_bgr_to_rgb(bgr_image: np.array) -> np.array:
    return bgr_image[:, :, ::-1]  # Reverses the third dimension (color channels)


def create_video_from_images(
    images: List[np.array], video_name: str = "episode_1", fps: int = 20
):
    # Convert each NumPy array image to an ImageClip
    clips = [ImageClip(convert_bgr_to_rgb(np_img.squeeze(0))) for np_img in images]

    # Set the duration of each clip to match the desired FPS
    # Note: This assumes all images should be displayed for an equal amount of time.
    for clip in clips:
        clip.duration = 1 / fps

    # Concatenate the ImageClips into a single video
    final_clip = concatenate_videoclips(clips, method="compose")

    # Write the result to a video file
    final_clip.write_videofile(video_name, fps=fps)