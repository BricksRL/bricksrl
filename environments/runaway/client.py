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

hub = InventorHub()
#hub.displaytext('ANDRE',on=1000,off=100)
#hub.display.text("{}".format(hub.battery.voltage()))
# Initialize the drive base.
left_motor = Motor(Port.E, Direction.COUNTERCLOCKWISE)
right_motor = Motor(Port.A)
drive_base = DriveBase(left_motor, right_motor, wheel_diameter=56, axle_track=130)
# Initialize the distance sensor.
sensor = UltrasonicSensor(Port.C)

# Optional: Register stdin for polling. This allows
# you to wait for incoming data without blocking.
keyboard = poll()
keyboard.register(stdin)


while True:

    # Optional: Check available input.
    while not keyboard.poll(0):
        # Optional: Do something here.
        wait(1)

    # Read action command
    action_cmd = stdin.buffer.read(4)
    # action_cmd = b'\xbf\xa0\x00\x00' # 1.25
    action_value = ustruct.unpack("!f", action_cmd)[0]
    
    action = action_value * 100
    sign = action_value // (abs(action_value) + 1e-12)

    if action_value > 0.0: # forward 
        drive_base.straight(abs(action))
    elif action_value < 0.0: # backward
        drive_base.straight(abs(action))
    else:
        raise ValueError("Invalid action value: {}".format(action_value))

    # get current state of the robot
    (left, right) = (left_motor.angle(), right_motor.angle())
    (pitch, roll) = hub.imu.tilt()
    dist = sensor.distance()

    # send current state
    out_msg = ustruct.pack('!fffff',left,right,pitch,roll,dist)
    stdout.buffer.write(out_msg)


