# SPDX-License-Identifier: MIT
# Copyright (c) 2020 Henrik Blidh
# Copyright (c) 2022 The Pybricks Authors

import asyncio
import struct
from bleak import BleakScanner, BleakClient

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
            data = struct.unpack('!iiiii',data)
        except:
            data = (0,0,0,0,0)

        await queue.put(data)

    try:
        # Connect and get services.
        print("Switch on the hub",flush=True)
        await client.connect()
        await client.start_notify(UART_TX_CHAR_UUID, handle_rx)
        nus = client.services.get_service(UART_SERVICE_UUID)
        rx_char = nus.get_characteristic(UART_RX_CHAR_UUID)

        # Tell user to start program on the hub.
        print("Start the program on the hub now with the button.",flush=True)
        await asyncio.sleep(5)

        # Send a few messages to the hub.
        for i in range(10):
            print(i)
            await send(client, b"a")
            data = await queue.get()
            print("Received: ", str(data))
            #await asyncio.sleep(1)
    

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


