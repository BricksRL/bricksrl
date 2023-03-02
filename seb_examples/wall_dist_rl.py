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
import time
from seb_examples.utils import logout, login, create_transition_td, handle_disconnect, data2numpy

UART_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
UART_RX_CHAR_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
UART_TX_CHAR_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

# Replace this with the name of your hub if you changed
# it when installing the Pybricks firmware.
HUB_NAME = "Pybricks Hub"

def hub_filter(device, ad):
    return device.name and device.name.lower() == HUB_NAME.lower()

def reward_function(state, action, next_state):
    """ Reward function for the wall distance task.
        Goal: to get away from the wall as fast as possible.
        
    """
    done = False
    
    if next_state[:, -1] <= 0.04: # too close to the wall break episode
        reward = -10
        done = True
    elif next_state[:, -1] < state[:, -1]:
        reward = -1.
    elif next_state[:, -1] > state[:, -1]:
        reward = 1.
    else:
        reward = 0.

    return np.array([reward]), np.array([done])


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
            data = struct.unpack("!fffff", data)
        except:
            data = (0.0, 0.0, 0.0, 0.0, 0.1)

        await queue.put(data)
    
    # Defines action and state space
    # Action dim = 2 as we only move forward or backward 
    # State dim = 1 as we only have one sensor: distance to the wall we try to reach up to ~ 10 cm
    action_dim = 1
    state_dim = 5 # 4 sensors (left,right,pitch,roll) + 1 distance to the wall
    action_space = BoundedTensorSpec(minimum=-torch.ones(action_dim), maximum=torch.ones(action_dim), shape=(action_dim,))
    states_space = BoundedTensorSpec(minimum=torch.zeros(state_dim), maximum=torch.ones(state_dim)*2000, shape=(state_dim,))
    
    # Create agent
    #agent = TD3Agent(action_space=action_space, state_space=states_space, learning_rate=3e-4, device="cpu")
    agent = SACAgent(action_space=action_space, state_space=states_space, learning_rate=3e-4, device="cpu")
    print("--- Agent initialized ---", flush=True)
    login(agent)
    
    # Initialize wandb
    wandb.init(project="lego-wall-td3", config=None) # TODO add config
    wandb.watch(agent.actor, log_freq=1)
    epochs = 100
    steps_per_epoch = 10
    
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
        
        input("Press Enter to start training...")
        for e in range(epochs):
            print("Start epoch: ", e)
            # get initial observation
            byte_action = struct.pack("!f", 0.001)
            await send(client, byte_action)    
            observation = data2numpy(await queue.get())
            # normalize observation
            observation = observation / 1000.
            
            done = np.array([False])
            rewards = 0
            for i in range(steps_per_epoch):
                start = time.time()
                print("Step: ", i)
                if e in [0]:
                    action = np.random.uniform(-1, 1, size=action_dim)
                    print("random action!")
                else:
                    action = agent.get_action(observation)
                    print("action from agent!")
                print("Action: ", action)
                # send action to hub
                byte_action = struct.pack("!f", action)
                await send(client, byte_action)
                
                next_observation = await queue.get()
                next_observation = data2numpy(next_observation)
                next_observation = next_observation / 1000. # normalize observation
                print("Received data: ", next_observation)
                reward, done = reward_function(observation, action, next_observation)
                rewards += reward
                if i == steps_per_epoch - 1:
                    done = np.array([True])
                transition = create_transition_td(observation, action, reward, next_observation, done)
                print("Past distance: ", observation[:, -1],
                      " | Current distance: ", next_observation[:, -1],
                      " | Reward: ", reward,
                      " | Done: ", done,
                      " | Buffer size: ", agent.replay_buffer.__len__())
                agent.add_experience(transition)
                observation = next_observation
                if done:
                    break
                print("Time for step: ", time.time() - start)
                # await asyncio.sleep(1)
            print("_"*50)
            print("\nFinished epoch: ", e, "Rewards: ", rewards)
            print("Training agent ...")
            # train agent
            loss = agent.train(batch_size=12, num_updates=5)
            print("Actor loss: ", loss["loss_actor"].item(), "Critic loss: ", loss["loss_qvalue"].item())
            print("\n")
            
            wandb.log({"epoch": e,
                       "reward": rewards,
                       "loss_actor": loss["loss_actor"].item(),
                       "loss_critic": loss["loss_qvalue"].item(),
                       "buffer_size": agent.replay_buffer.__len__(),
                       "final_distance": observation[:, -1]})
            
            
            reset_robot = input("Reset robot to initial position? (y/n): ")
            if reset_robot == "y":
                pass
            else:
                print("Stopping training ...")
                logout(agent)
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
