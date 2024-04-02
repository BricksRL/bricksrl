import umath
import urandom
import ustruct
from micropython import kbd_intr
from pybricks.hubs import InventorHub
from pybricks.parameters import Port
from pybricks.pupdevices import ColorSensor, Motor
from pybricks.tools import wait
from uselect import poll

# Standard MicroPython modules
from usys import stdin, stdout

kbd_intr(-1)

hub = InventorHub()

# Initialize the drive base.
# Grab Motor range (130, 179) left side closed (-148, -45)
grab_motor_range = (-148, -45)
grab_motor = Motor(Port.E)
grab_motor.run_target(speed=400, target_angle=-95)  # start roughly in the middle
# High Motor range (-150, 30)
high_motor_range = (-150, 10)
high_motor = Motor(Port.A)
high_motor.run_target(speed=400, target_angle=-70)

# Low motor range (10, 70)
low_motor_range = (10, 70)
low_motor = Motor(Port.D)
low_motor.control.limits(500, 1000, 900)
low_motor.run_target(speed=400, target_angle=40)

# Rotation motor range (-360, 360)
# observe as its basically ~ 180
rotation_motor = Motor(Port.B)

# color_sensor = ColorSensor(Port.C)
motors = {"GM": grab_motor, "HM": high_motor, "LM": low_motor, "RM": rotation_motor}


def get_current_motor_angles():
    angles = {}
    for k, v in motors.items():
        angle = normalize_angle(get_angle(v))
        angles.update({k: angle})
    return angles


def run_angle(motor, angle, speed=300):
    motor.run_angle(speed=speed, rotation_angle=angle, wait=False)


def get_angle(motor):
    return motor.angle()


def normalize_angle(angle, low_angle=-180, high_angle=179, original_one_round=360):
    # Normalize angle to be within -179 to 179 degrees
    while angle <= low_angle:
        angle += original_one_round
    while angle > high_angle:
        angle -= original_one_round
    return angle


def transform_range(value, old_min, old_max, new_min, new_max):
    """
    Transform a value from one range to another.

    Parameters:
    value (float): The value to transform.
    old_min (float): The minimum value of the old range.
    old_max (float): The maximum value of the old range.
    new_min (float): The minimum value of the new range.
    new_max (float): The maximum value of the new range.

    Returns:
    float: The transformed value.
    """
    # Compute the scale factor between the old and new ranges
    scale = (new_max - new_min) / (old_max - old_min)
    # Apply the transformation
    return new_min + (value - old_min) * scale


keyboard = poll()
keyboard.register(stdin)

while True:

    while not keyboard.poll(0):
        wait(1)

    # Read action values for the motors
    data = stdin.buffer.read(16)  # Reading 4 bytes (4 floats)
    rotation_action, low_action, high_action, grab_action = ustruct.unpack(
        "!ffff", data
    )

    # transform action range for motors
    grab_action = transform_range(grab_action, -1, 1, -25, 25)
    high_action = transform_range(high_action, -1, 1, -60, 60)
    low_action = transform_range(low_action, -1, 1, -30, 30)
    rotation_action = transform_range(rotation_action, -1, 1, -180, 180)

    angles = get_current_motor_angles()

    if not (angles["GM"] + grab_action > max(grab_motor_range)) and not (
        angles["GM"] + grab_action < min(grab_motor_range)
    ):
        grab_motor.run_angle(speed=250, rotation_angle=grab_action, wait=False)

    if not (angles["HM"] + high_action > max(high_motor_range)) and not (
        angles["HM"] + high_action < min(high_motor_range)
    ):
        high_motor.run_angle(speed=250, rotation_angle=high_action, wait=False)

    if not (angles["LM"] + low_action > max(low_motor_range)) and not (
        angles["LM"] + low_action < min(low_motor_range)
    ):
        low_motor.run_angle(speed=250, rotation_angle=low_action, wait=False)

    # if not (angles["RM"] + rotation_action > 180) or not (angles["RM"] + rotation_action < -180):
    rotation_motor.run_angle(speed=250, rotation_angle=rotation_action, wait=False)

    wait(250)

    rotation_angle = rotation_motor.angle()
    high_angle = high_motor.angle()
    grab_angle = grab_motor.angle()
    low_angle = low_motor.angle()

    # GM HM LM RM
    out_msg = ustruct.pack(
        "!ffff",
        grab_angle,
        low_angle,
        normalize_angle(high_angle),
        normalize_angle(
            rotation_angle, low_angle=-900, high_angle=900, original_one_round=1800
        ),
    )
    stdout.buffer.write(out_msg)
