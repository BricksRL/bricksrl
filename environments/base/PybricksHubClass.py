import asyncio
import struct

from bleak import BleakClient, BleakScanner


class PybricksHub:
    """Class for connecting to a Pybricks Hub."""

    PYBRICKS_COMMAND_EVENT_CHAR_UUID = "c5f50002-8280-46da-89f4-6d8051e4aeef"
    HUB_NAME = "Pybricks Hub"

    def __init__(self, out_format_str: str, state_dim: int):
        self.device = None
        self.client = None
        self.rx_queue = asyncio.Queue(maxsize=8)  # LifoQueue
        self.loop = asyncio.get_event_loop()

        self.exception_out_data = struct.pack(out_format_str, *([0.0] * state_dim))
        self.disconnected = False
        self.payload_buffer = None

    def connect(self) -> None:
        """Connect to the hub."""
        print("Connecting to the hub...")
        self.loop.run_until_complete(self._connect())


    async def _connect(self) -> None:
        """Connect to the hub."""
        try:
            # Find the device and initialize client.
            self.device = await BleakScanner.find_device_by_name(self.HUB_NAME)
            self.client = BleakClient(
                self.device, disconnected_callback=self._handle_disconnect
            )

            # Connect and get services
            print("Switch on the hub", flush=True)
            await self.client.connect()
            await self.client.start_notify(
                self.PYBRICKS_COMMAND_EVENT_CHAR_UUID, self._handle_rx
            )

            # Tell user to start program on the hub.
            print("Start the program on the hub now with the button.", flush=True)
            await asyncio.sleep(5)

        except Exception as e:
            # Handle exceptions.
            print(e)
            await self.disconnect()

    def send(self, data: bytes) -> None:
        """Send data to the hub as bytes."""
        self.loop.run_until_complete(self._send(data))

    async def _send(self, data: bytes) -> None:
        try:
            # Send some data to the hub.
            await self.client.write_gatt_char(
                self.PYBRICKS_COMMAND_EVENT_CHAR_UUID,
                b"\x06" + data,  # Prepend "write stdin" command b"\x06" + data
                response=False, # True / false?
            )
        except Exception as e:
            # Handle exceptions.
            print(e)
            await self.disconnect()


    def disconnect(self) -> None:
        """
        Disconnect from the hub.
        This method disconnects the hub from the client.
        """
        if self.client and not self.disconnected:
            asyncio.create_task(self._disconnect())


    async def _disconnect(self) -> None:
        try:
            # Disconnect when we are done.
            if self.client:
                await self.client.disconnect()
        except Exception as e:
            # Handle exceptions.
            print(e)
        finally:
            self.disconnected = True
        
        self.client = None
        self.device = None
        self.rx_char = None

    def _handle_disconnect(self, _) -> None:
        print("Hub was disconnected.")
        self.disconnect()

    async def _handle_rx(self, _, data: bytes) -> None:
        # add received data to the queue
        if data[0] == 0x01:  # "write stdout" event (0x01)
            payload = data[1:]
            print("Received:", payload)
            if len(payload) != self.exception_out_data and self.payload_buffer is None:
                self.payload_buffer = payload
            elif len(payload) != self.exception_out_data and self.payload_buffer is not None:
                self.payload_buffer += payload
                if self.payload_buffer.__len__() == len(self.exception_out_data):
                    await self.rx_queue.put(self.payload_buffer)
                    self.payload_buffer = None
            else:
                await self.rx_queue.put(payload)


    async def _read_data(self) -> bytes:
        try:
            # get data from the queue
            return await self.rx_queue.get()
        except asyncio.QueueEmpty:
            print("Queue is empty, returning zeros")
            return self.exception_out_data

    def read(self) -> bytes:
        """Read data from the hub and return it as a bytearray."""
        return self.loop.run_until_complete(self._read_data())

    def close(self) -> None:
        if not self.loop.is_closed():
            self.loop.run_until_complete(self._disconnect())
