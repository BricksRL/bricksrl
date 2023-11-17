import numpy as np
import tensordict as td
import torch


def data2numpy(data: list):
    """Convert a list of bytes to a numpy array."""
    return np.array(data)[None, :]


def handle_disconnect(_):
    print("Hub was disconnected.")


def tensordict2dict(td: td.TensorDict) -> dict:
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
        save_name = input("Enter the name of the file to save: ")
        torch.save(buffer_dict, save_name + ".pth")



def login(agent):
    x = input("Do you want to load the model? (y/n)")
    if x == "y":
        agent.load_model("lego-wall-td3")
    else:
        print("Model not loaded!")
    x = input("Do you want to load the replay buffer? (y/n)")
    if x == "y":
        save_name = input("Enter the name of the file to load: ")
        agent.load_replaybuffer(save_name)
    else:
        print("Buffer not loaded!")


def prefill_buffer(env, agent, num_episodes):
    """
    Prefills the agent's replay buffer with experiences by running the environment for a specified number of episodes.
    
    Args:
    - env: gym.Env object representing the environment
    - agent: Agent object with an add_experience method to add experiences to the replay buffer
    - num_episodes: int, number of episodes to run the environment for
    
    Returns: None
    """
    if agent.name in ["sac", "td3"]:
        for e in range(num_episodes):
            inpt = input("Press Enter to start prefilling episode: ")
            print("Prefill episode: ", e)
            state = env.reset()
            done = False
            while not done:
                action = np.random.uniform(-1, 1, size=agent.action_space.shape[0])
                print("Random action: ", action)
                next_state, reward, done, info = env.step(action)
                transition = create_transition_td(
                    state, action, np.array([reward]), next_state, np.array([done])
                )
                agent.add_experience(transition)
                state = next_state
        print("Prefill done! Buffer size: ", agent.replay_buffer.__len__())
