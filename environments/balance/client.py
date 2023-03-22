# NOTE: Run this program with the latest
# firmware provided via https://beta.pybricks.com/

from pybricks.hubs import InventorHub
from pybricks.pupdevices import Motor
from pybricks.parameters import Port, Direction
from pybricks.tools import wait
from pybricks.robotics import DriveBase

# Standard MicroPython modules
from usys import stdin, stdout
from uselect import poll
import ustruct
from micropython import kbd_intr

hub = InventorHub()

# Initialize the drive base.
left_motor = Motor(Port.E, Direction.COUNTERCLOCKWISE)
right_motor = Motor(Port.A)
drive_base = DriveBase(left_motor, right_motor, wheel_diameter=56, axle_track=130)

# Optional: Register stdin for polling. This allows
# you to wait for incoming data without blocking.
keyboard = poll()
keyboard.register(stdin)
# ignore keyboard interrupt
kbd_intr(-1)
# stop button pressed could be due to low energy

while True:

    # Optional: Check available input.
    while not keyboard.poll(0):
        wait(1)

    action_value = ustruct.unpack("!f", stdin.buffer.read(4))[0]
    action = action_value * 100
    if type(action) == float:
        pass
    else:
        action = 0.0
    drive_base.straight(action, wait=False)

    wait(50) #800 seems to work 

    # get current state of the robot
    (left, right) = (left_motor.angle(), right_motor.angle())
    (pitch, roll) = hub.imu.tilt()

    # send current state
    out_msg = ustruct.pack('!ffff',left, right, pitch, roll)
    stdout.buffer.write(out_msg)
