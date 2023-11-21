# NOTE: Run this program with the latest
# firmware provided via https://beta.pybricks.com/

import ustruct
from micropython import kbd_intr
from pybricks.hubs import InventorHub
from pybricks.parameters import Direction, Port
from pybricks.pupdevices import ColorSensor, Motor, UltrasonicSensor
from pybricks.robotics import DriveBase
from pybricks.tools import wait
from uselect import poll

# Standard MicroPython modules
from usys import stdin, stdout

kbd_intr(-1)

hub = InventorHub()

# Initialize the drive base.
lf_motor = Motor(Port.D, Direction.COUNTERCLOCKWISE)
lb_motor = Motor(Port.B, Direction.COUNTERCLOCKWISE)
rf_motor = Motor(Port.C)
rb_motor = Motor(Port.A)

color = ColorSensor(Port.F)
eyes = UltrasonicSensor(Port.E)

# target angle: 179 is good starting position!
keyboard = poll()
keyboard.register(stdin)


while True:
    # Optional: Check available input.
    while not keyboard.poll(0):
        wait(1)

    # Read action values for both motors
    data = stdin.buffer.read(16)  # Reading 16 bytes (4 floats)
    lf_value, lb_value, rf_value, rb_value = ustruct.unpack("!ffff", data)

    # set motor angle. Range is [-180, 179] action outputs are [-1, 1] -> we multiply by 179
    scaling_factor = 179

    lf_motor.track_target(target_angle=lf_value * scaling_factor)
    lb_motor.track_target(target_angle=lb_value * scaling_factor)
    rf_motor.track_target(target_angle=rf_value * scaling_factor)
    rb_motor.track_target(target_angle=rb_value * scaling_factor)

    wait(50)  # Small delay

    # get current state of the robot
    (lf_angle, rf_angle) = (lf_motor.angle(), rf_motor.angle())
    (lb_angle, rb_angle) = (lb_motor.angle(), rb_motor.angle())
    (pitch, roll) = hub.imu.tilt()
    dist = eyes.distance()

    # send current state
    out_msg = ustruct.pack(
        "!fffff", lf_angle, rf_angle, lb_angle, rb_angle, dist #pitch, roll,
    )
    stdout.buffer.write(out_msg)
