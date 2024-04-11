import umath
import urandom
import ustruct
from micropython import kbd_intr
from pybricks.hubs import InventorHub
from pybricks.parameters import Port
from pybricks.pupdevices import Motor
from pybricks.tools import wait
from uselect import poll

# Standard MicroPython modules
from usys import stdin, stdout

kbd_intr(-1)

hub = InventorHub()

# High Motor range (-150, 30)
high_motor_range = (-150, 10)
high_motor = Motor(Port.A)
high_motor.run_target(speed=400, target_angle=-70)

# Low motor range (0, 120)
low_motor_range = (10, 70)
low_motor = Motor(Port.D)
low_motor.control.limits(500, 1000, 900)
low_motor.run_target(speed=200, target_angle=40)

# Rotation motor range (-20, 160)
rotation_motor_range = (-140, 40)
rotation_motor = Motor(Port.B, gears=[20, 60])
print(rotation_motor.angle())
motors = {"HM": high_motor, "LM": low_motor, "RM": rotation_motor}


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
    data = stdin.buffer.read(12)
    rotation_action, low_action, high_action = ustruct.unpack("!fff", data)

    # transform action range for motors
    high_action = transform_range(high_action, -1, 1, -60, 60)
    low_action = transform_range(low_action, -1, 1, -30, 30)
    rotation_action = transform_range(rotation_action, -1, 1, -90, 90)

    angles = get_current_motor_angles()

    # Adjust high action to ensure it stays within range after being applied
    if angles["HM"] + high_action > max(high_motor_range):
        high_action = max(high_motor_range) - angles["HM"]
    elif angles["HM"] + high_action < min(high_motor_range):
        high_action = min(high_motor_range) - angles["HM"]
    high_motor.run_angle(speed=250, rotation_angle=high_action, wait=False)

    # Adjust low action to ensure it stays within range after being applied
    if angles["LM"] + low_action > max(low_motor_range):
        low_action = max(low_motor_range) - angles["LM"]
    elif angles["LM"] + low_action < min(low_motor_range):
        low_action = min(low_motor_range) - angles["LM"]
    low_motor.run_angle(speed=250, rotation_angle=low_action, wait=False)

    # Adjust rotation action to ensure it stays within range after being applied
    if angles["RM"] + rotation_action > max(rotation_motor_range):
        rotation_action = max(rotation_motor_range) - angles["RM"]
    elif angles["RM"] + rotation_action < min(rotation_motor_range):
        rotation_action = min(rotation_motor_range) - angles["RM"]
    rotation_motor.run_angle(speed=250, rotation_angle=rotation_action, wait=False)
    wait(250)

    high_angle = high_motor.angle()
    low_angle = low_motor.angle()
    rotation_angle = rotation_motor.angle()

    # sometimes low angle jumps out of range and cant move back this corrects those cases
    if low_angle < 10:
        low_motor.run_target(speed=200, target_angle=10)

    # HM LM RM
    out_msg = ustruct.pack(
        "!fff",
        high_angle,
        low_angle,
        rotation_angle,
    )
    stdout.buffer.write(out_msg)
