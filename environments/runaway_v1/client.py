# NOTE: Run this program with the latest
# firmware provided via https://beta.pybricks.com/

from pybricks.hubs import InventorHub
from pybricks.pupdevices import Motor, UltrasonicSensor
from pybricks.parameters import Port, Direction
from pybricks.tools import wait
from pybricks.robotics import DriveBase

# Standard MicroPython modules
from usys import stdin, stdout
from uselect import poll
import ustruct
from micropython import kbd_intr

kbd_intr(-1)

hub = InventorHub()

# Initialize the drive base.
left_motor = Motor(Port.E, Direction.COUNTERCLOCKWISE)
right_motor = Motor(Port.A)
#drive_base = DriveBase(left_motor, right_motor, wheel_diameter=56, axle_track=130)
# Initialize the distance sensor.
sensor = UltrasonicSensor(Port.C)

# Optional: Register stdin for polling. This allows
# you to wait for incoming data without blocking.
keyboard = poll()
keyboard.register(stdin)

while True:

    # Optional: Check available input.
    while not keyboard.poll(0):
        wait(1)

    # Read action values for both motors
    data = stdin.buffer.read(8)  # Reading 8 bytes (two floats)
    left_action_value, right_action_value = ustruct.unpack("!ff", data)
    if type(left_action_value) == float and type(right_action_value) == float:
        pass
    else:
        action = 0.0
    # Apply action to each motor

    left_motor.run_angle(speed=200, rotation_angle=left_action_value * 100, wait=False)
    right_motor.run_angle(speed=200, rotation_angle=right_action_value * 100, wait=False)

    wait(50)  # Small delay 

    # get current state of the robot
    (left, right) = (left_motor.angle(), right_motor.angle())
    (pitch, roll) = hub.imu.tilt()
    dist = sensor.distance()

    # send current state
    out_msg = ustruct.pack('!fffff',left, right, pitch, roll, dist)
    stdout.buffer.write(out_msg)
