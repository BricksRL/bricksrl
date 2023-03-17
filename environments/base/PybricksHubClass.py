import asyncio
from bleak import BleakScanner, BleakClient
from bleak.backends.device import BLEDevice
import struct


class PybricksHub:
    """ Class for connecting to a Pybricks Hub. """
    UART_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
    UART_RX_CHAR_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
    UART_TX_CHAR_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
    HUB_NAME = "Pybricks Hub"

    def __init__(self, out_format_str: str, state_dim: int):
        self.device = None
        self.client = None
        self.rx_char = None
        self.rx_queue = asyncio.LifoQueue(maxsize=5)# LifoQueue
        self.loop = asyncio.get_event_loop()
        # asyncio.set_event_loop(self.loop)
        
        self.exception_out_data = struct.pack(out_format_str, *([0.0] * state_dim))
        
    def connect(self)-> None:
        """ Connect to the hub. """
        print("Connecting to the hub...")
        self.loop.run_until_complete(self._connect())

    async def _connect(self)-> None:
        """ Connect to the hub. """
        try:
            # Find the device and initialize client.
            self.device = await BleakScanner.find_device_by_filter(self._hub_filter)
            self.client = BleakClient(self.device, disconnected_callback=self._handle_disconnect)

            # Connect and get services
            print("Switch on the hub", flush=True)
            await self.client.connect()
            await self.client.start_notify(self.UART_TX_CHAR_UUID, self._handle_rx)
            nus = self.client.services.get_service(self.UART_SERVICE_UUID)
            self.rx_char = nus.get_characteristic(self.UART_RX_CHAR_UUID)

            # Tell user to start program on the hub.
            print("Start the program on the hub now with the button.", flush=True)
            await asyncio.sleep(5)

        except Exception as e:
            # Handle exceptions.
            print(e)
            await self.disconnect()

    def send(self, data: bytes)-> None:
        """ Send data to the hub as bytes. """
        self.loop.run_until_complete(self._send(data))


    async def _send(self, data: bytes)-> None:
        try:
            # Send some data to the hub.
            await self.client.write_gatt_char(self.rx_char, data)
        except Exception as e:
            # Handle exceptions.
            print(e)
            await self.disconnect()

    def disconnect(self)-> None:
        self.loop.run_until_complete(self._disconnect())
        self.loop.close()

    async def _disconnect(self)-> None:
        try:
            # Disconnect when we are done.
            if self.client:
                await self.client.disconnect()
                self.client = None
                self.device = None
                self.rx_char = None
        except Exception as e:
            # Handle exceptions.
            print(e)
        finally:
            self.disconnected = True


    @staticmethod
    def _hub_filter(device: BLEDevice, ad)-> None:
        return device.name and device.name.lower() == PybricksHub.HUB_NAME.lower()

    def _handle_disconnect(self, _)-> None:
        print("Hub was disconnected.")
        self.disconnect()

    async def _handle_rx(self, _, data: bytes)-> None:
        # add received data to the queue
        await self.rx_queue.put(data)
    
    async def _read_data(self)-> bytes:
        try:
            # get data from the queue
            return await self.rx_queue.get()
        except asyncio.QueueEmpty:
            print("Queue is empty, returning zeros")
            return self.exception_out_data

    
    def read(self)-> bytes:
        """Read data from the hub and return it as a bytearray."""
        # try:
        #     # get data from the queue
        #     return self.rx_queue.get_nowait()
        # except asyncio.QueueEmpty:
        #     # TODO: handle exceptions currently just return zeros
        #     print("Queue is empty, returning zeros")
        #     return self.exception_out_data #struct.pack("!fffff", 0.0, 0.0, 0.0, 0.0, 0.0)
        return self.loop.run_until_complete(self._read_data())


    def close(self)-> None:
        # TODO: this is not working correctly -- "never awaited disconnect"
        # self.disconnect()
        # Disconnect from the hub.
        self.loop.run_until_complete(self._disconnect())

        # Wait for the disconnection to complete.
        timeout = 5  # seconds
        start_time = self.loop.time()
        while not self.disconnected:
            self.loop.run_until_complete(asyncio.sleep(0.1))
            if self.loop.time() - start_time > timeout:
                print("Timed out waiting for disconnection to complete.")
                break

        # Close the event loop.
        self.loop.close()

    
# # Create an instance of the PybricksHub class.
# hub = PybricksHub()

# # Connect to the hub.
# hub.connect()

# # Send a few messages to the hub.
# for i in range(5):
#     byte_action = struct.pack("!f", -0.5)
#     hub.send(byte_action)
#     time.sleep(1)
#     data = hub.read()
#     print("Reading data: ", struct.unpack("!fffff", data))
#     time.sleep(1)
#     byte_action = struct.pack("!f", 0.5)
#     hub.send(byte_action)
#     time.sleep(1)
