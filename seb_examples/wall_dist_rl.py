# SPDX-License-Identifier: MIT
# Copyright (c) 2020 Henrik Blidh
# Copyright (c) 2022 The Pybricks Authors

import asyncio
import struct
from bleak import BleakScanner, BleakClient
import torch
import numpy as np
from torchrl.data import BoundedTensorSpec
import tensordict as td

from agents import TD3Agent, SACAgent
import wandb

UART_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
UART_RX_CHAR_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
UART_TX_CHAR_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

# Replace this with the name of your hub if you changed
# it when installing the Pybricks firmware.
HUB_NAME = "Pybricks Hub"


def hub_filter(device, ad):
    return device.name and device.name.lower() == HUB_NAME.lower()

def handle_disconnect(_):
    print("Hub was disconnected.")

def mse(x, y):
    """ Calculate the mean squared error between two numpy arrays. """
    return np.mean((x - y) ** 2)

def create_transition_td(observation: np.array,
                         action: np.array,
                         reward: np.array,
                         next_observation: np.array,
                         done: np.array,
                         batch_size: int=1):
    """ Create a TensorDict from a transition tuple. """
    obs_t = torch.from_numpy(observation).float()
    action_t = torch.from_numpy(action).float()[None, :]
    reward_t = torch.from_numpy(reward).float()[None, :]
    next_obs_t = torch.from_numpy(next_observation).float()
    done_t = torch.from_numpy(done).bool()[None, :]

    return td.TensorDict({
        "observation": obs_t,
        "action": action_t,
        "reward": reward_t,
        "next": {"observation": next_obs_t},
        "done": done_t} , batch_size=batch_size
    )

def data2numpy(data: list):
    """ Convert a list of bytes to a numpy array. """
    return np.array(data)[None, :]

def get_distance(observation, goal_min_dist=100):
    """ Calculate the mse error between the robot and the wall. """
    return mse(observation[:, -1], goal_min_dist)

def get_abs_distance(observation, goal_min_dist=100):
    """ Calculate the absolute error between the robot and the wall. """
    distance = np.abs(observation[:, -1] - goal_min_dist)
    return distance
    
def reward_function(state, past_distance=2000):
    possible_done = np.array([False])
    goal_min_dist = 100 # mm
    # mse between current distance and goal distance
    current_distance = get_abs_distance(state, goal_min_dist)
    threshold_delta = 5 # mm
    if current_distance < past_distance and abs(past_distance - current_distance) >= threshold_delta:
        # reward = 0.5
        reward = float(abs(current_distance - past_distance)) / 10
    elif current_distance == past_distance:
        reward = 0.
    elif current_distance > past_distance and abs(past_distance - current_distance) >= threshold_delta:
        # reward = -0.5
        reward = - float(abs(current_distance - past_distance)) / 10 
    else:
        reward = 0.
    if state[:, -1] <= 40:
        reward = -10.
        possible_done = np.array([True])
    if current_distance >= 10 and current_distance <= 10:
        reward = 100.
        possible_done = np.array([True])
    if state[:, -1] > 500:
        reward = -10.
        possible_done = np.array([True])    
        
    return np.array([reward]), current_distance, possible_done

def discretize(value):
    index = round(value.item() * 10)
    return index

def get_discrete_action(index):
    vocab =  {10: "a", 9: "b", 8: "c", 7: "d", 6: "e", 5: "f", 4: "g", 3: "h", 2: "i", 1: "j", 0: "k", # forward movements 100 to 0
              -1: "l", -2: "m", -3: "n", -4: "o", -5: "p", -6: "q", -7: "r", -8: "s", -9: "t", -10: "u"} # backward movements -10 to -100
    return vocab[index]

async def main():
    # Find the device and initialize client.
    device = await BleakScanner.find_device_by_filter(hub_filter)
    client = BleakClient(device, disconnected_callback=handle_disconnect)
    queue = asyncio.LifoQueue()

    # Shorthand for sending some data to the hub.
    async def send(client, data):
        await client.write_gatt_char(rx_char, data)

    async def handle_rx(_, data: bytearray):
        try:
            data = struct.unpack('!iiiii', data)
        except:
            data = (0,0,0,0,0)

        await queue.put(data)
    
    # Defines action and state space
    # Action dim = 2 as we only move forward or backward 
    # State dim = 1 as we only have one sensor: distance to the wall we try to reach up to ~ 10 cm
    action_dim = 1
    state_dim = 5 # 4 sensors (left,right,pitch,roll) + 1 distance to the wall
    action_space = BoundedTensorSpec(minimum=-torch.ones(action_dim), maximum=torch.ones(action_dim), shape=(action_dim,))
    states_space = BoundedTensorSpec(minimum=torch.zeros(state_dim), maximum=torch.ones(state_dim)*2000, shape=(state_dim,))
    
    # Create agent
    agent = TD3Agent(action_space=action_space, state_space=states_space, learning_rate=1e-5, device="cpu")
    print("--- Agent initialized ---", flush=True)
    
    # Initialize wandb
    wandb.init(project="lego-wall-td3", config=None) # TODO add config
    wandb.watch(agent.actor)
    epochs = 100
    steps_per_epoch = 20
    
    try:
        # Connect and get services.
        print("Switch on the hub", flush=True)
        await client.connect()
        await client.start_notify(UART_TX_CHAR_UUID, handle_rx)
        nus = client.services.get_service(UART_SERVICE_UUID)
        rx_char = nus.get_characteristic(UART_RX_CHAR_UUID)

        # Tell user to start program on the hub.
        print("Start the program on the hub now with the button.", flush=True)
        await asyncio.sleep(5)
        
        for e in range(epochs):
            print("Start epoch: ", e)
            await send(client, b"k") # send initial message to start the program and be able to receive initial obs [k-> do nothing]    
            observation = await queue.get()
            observation = data2numpy(observation)
            print("Inital data: ", observation)
            past_distance = get_abs_distance(observation)
            done = np.array([False])
            rewards = 0
            for i in range(steps_per_epoch):

                print("Step: ", i)
                conti_action = agent.get_action(observation)
                print("Agent action: ", conti_action)
                action_message = get_discrete_action(discretize(conti_action))
                print("Action message: ", action_message)
                message_as_bytes = action_message.encode("utf-8")
                print("Message as bytes: ", message_as_bytes)
                await send(client, message_as_bytes)
                
                next_observation = await queue.get()
                next_observation = data2numpy(next_observation)
                print("Received data: ", next_observation)
                reward, current_distance, done = reward_function(next_observation, past_distance)
                rewards += reward
                if i == steps_per_epoch - 1:
                    done = np.array([True])
                transition = create_transition_td(observation, conti_action, reward, next_observation, done)
                print("Past distance: ", past_distance,
                      " | Current distance: ", current_distance,
                      " | Reward: ", reward,
                      " | Done: ", done,
                      " | Buffer size: ", agent.replay_buffer.__len__())
                agent.add_experience(transition)
                past_distance = current_distance
                if done:
                    break
                
                # await asyncio.sleep(1)
            print("_"*50)
            print("\nFinished epoch: ", e, "Rewards: ", rewards)
            print("Training agent ...")
            # train agent
            loss = agent.train(batch_size=5, num_updates=5)
            print("Actor loss: ", loss["loss_actor"].item(), "Critic loss: ", loss["loss_qvalue"].item())
            print("\n")
            
            wandb.log({"epoch": e,
                       "reward": rewards,
                       "loss_actor": loss["loss_actor"].item(),
                       "loss_critic": loss["loss_qvalue"].item(),
                       "buffer_size": agent.replay_buffer.__len__(),
                       "final_distance": current_distance})
            
            
            reset_robot = input("Reset robot to initial position? (y/n): ")
            if reset_robot == "y":
                pass
            else:
                print("Stopping training ...")
                # TODO save model or training data
                break
                
        # Send a message to indicate stop.
        await send(client, b"b")
        await asyncio.sleep(1)

    except Exception as e:
        # Handle exceptions.
        print(e)
    finally:
        # Disconnect when we are done.
        await client.disconnect()


# Run the main async program.
asyncio.run(main())


