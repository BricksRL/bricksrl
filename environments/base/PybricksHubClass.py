import asyncio
import struct

from bleak import BleakClient, BleakScanner
from bleak.backends.device import BLEDevice


class PybricksHub:
    """Class for connecting to a Pybricks Hub."""

    UART_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
    UART_RX_CHAR_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
    UART_TX_CHAR_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
    HUB_NAME = "Pybricks Hub"

    def __init__(self, out_format_str: str, state_dim: int):
        """
        Initialize the PybricksHub object.

        Args:
            out_format_str (str): The output format string.
            state_dim (int): The dimension of the state.

        Returns:
            None
        """
        self.device = None
        self.client = None
        self.rx_char = None
        self.rx_queue = asyncio.Queue(maxsize=5)  # LifoQueue
        self.loop = asyncio.get_event_loop()

        self.disconnected = False
        self.exception_out_data = struct.pack(out_format_str, *([0.0] * state_dim))

    def connect(self) -> None:
        """Connect to the hub."""
        print("Connecting to the hub...")
        self.loop.run_until_complete(self._connect())

    async def _connect(self) -> None:
        """
        Connect to the hub.

        Raises:
            Exception: If an error occurs during the connection process.
        """
        try:
            # Find the device and initialize client.
            self.device = await BleakScanner.find_device_by_filter(self._hub_filter)
            self.client = BleakClient(
                self.device, disconnected_callback=self._handle_disconnect
            )

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

    def send(self, data: bytes) -> None:
        """
        Send data to the hub as bytes.

        Args:
            data (bytes): The data to be sent.

        Returns:
            None
        """
        self.loop.run_until_complete(self._send(data))

    async def _send(self, data: bytes) -> None:
        """
        Send data to the hub as bytes.

        Args:
            data (bytes): The data to be sent.

        Returns:
            None
        """
        try:
            # Send some data to the hub.
            await self.client.write_gatt_char(self.rx_char, data)
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
        """Disconnect from the hub.

        This method disconnects the hub by calling the `disconnect()` method of the client.
        If an exception occurs during the disconnection process, it will be printed.
        Finally, the disconnected flag is set to True and the client, device, and rx_char attributes are reset to None.

        """
        try:
            if self.client:
                await self.client.disconnect()
        except Exception as e:
            print(e)
        finally:
            self.disconnected = True
            self.client = None
            self.device = None
            self.rx_char = None

    @staticmethod
    def _hub_filter(device: BLEDevice, ad) -> None:
        """
        Filter function to find the Pybricks Hub device.

        Args:
            device (BLEDevice): The BLE device.
            ad: The advertisement data.

        Returns:
            None
        """
        return device.name and device.name.lower() == PybricksHub.HUB_NAME.lower()

    def _handle_disconnect(self, _) -> None:
        """Handle hub disconnection."""
        print("Hub was disconnected!")
        self.disconnect()

    async def _handle_rx(self, _, data: bytes) -> None:
        """
        Handle received data from the hub.

        Args:
            _: The sender of the notification.
            data (bytes): The received data.

        Returns:
            None
        """
        # add received data to the queue
        await self.rx_queue.put(data)

    async def _read_data(self) -> bytes:
        """
        Reads data from the queue and returns it.

        If the queue is empty, it returns the exception_out_data.

        Returns:
            bytes: The data read from the queue or exception_out_data if the queue is empty.
        """
        try:
            # get data from the queue
            return await self.rx_queue.get()
        except asyncio.QueueEmpty:
            print("Queue is empty, returning zeros")
            return self.exception_out_data

    def read(self) -> bytes:
        """
        Read data from the hub and return it as a bytearray.

        Returns:
            bytes: The data read from the hub.
        """
        return self.loop.run_until_complete(self._read_data())

    def close(self) -> None:
        """
        Closes the connection to the Pybricks Hub.

        This method checks if the event loop is closed and then runs the `_disconnect` method asynchronously.

        Returns:
            None
        """
        if not self.loop.is_closed():
            self.loop.run_until_complete(self._disconnect())

