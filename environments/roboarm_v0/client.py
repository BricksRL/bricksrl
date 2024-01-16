# NOTE: Run this program with the latest
# firmware provided via https://beta.pybricks.com/

from pybricks.hubs import InventorHub
from pybricks.pupdevices import Motor
from pybricks.parameters import Port, Direction
from pybricks.tools import wait, StopWatch
from pybricks.parameters import Axis
import umath
# Standard MicroPython modules
from usys import stdin, stdout
from uselect import poll
import ustruct
from micropython import kbd_intr
import urandom

keyboard = poll()
keyboard.register(stdin)

hub = InventorHub()

# Initialize the drive base.
# Grab Motor range (-230, -148) left side closed (-148, -44)
grab_motor = Motor(Port.E)
# High Motor range (-60, 140)
high_motor = Motor(Port.A)

# Low motor range (-190, -10)
low_motor = Motor(Port.D)

# Rotation motor range (-360, 360) 
# observe as its basically ~ 180
rotation_motor = Motor(Port.B)

motors = {"GB": grab_motor, "HM": high_motor, "LM": low_motor, "RM": rotation_motor}

def get_current_motor_angles():
    angles = {}
    for k, v in motors.items():
        angle = get_angle(v)
        angles.update({k: angle})
    return angles

def run_angle(motor, angle, speed=300):
    motor.run_angle(speed=speed, rotation_angle=angle, wait=False)

def get_angle(motor):
    return motor.angle()

def normalize_angle(angle):
    # Normalize angle to be within -179 to 179 degrees
    while angle <= -180:
        angle += 360
    while angle > 179:
        angle -= 360
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

# set starting position
motor_speed = 500
grab_motor.run_target(speed=motor_speed, target_angle=-148)
high_motor.run_target(speed=motor_speed, target_angle=40)
low_motor.run_target(speed=motor_speed, target_angle=-90)
rotation_motor.run_target(speed=motor_speed, target_angle=0)


low_action_range = -100
high_action_range = 100

print("Battery: ", hub.battery.current())
while True:

    while not keyboard.poll(0):
        wait(1)

    # Read action values for both motors
    data = stdin.buffer.read(16)  # Reading 16 bytes (4 floats)

    grab_action, high_action, low_action, rotation_action = ustruct.unpack("!ffff", data)
    
    # transform action range for motors
    grab_action = transform_range(grab_action, -1, 1, low_action_range, high_action_range)
    high_action = transform_range(high_action, -1, 1, low_action_range, high_action_range)
    low_action = transform_range(low_action, -1, 1, low_action_range, high_action_range)
    rotation_action = transform_range(rotation_action, -1, 1, low_action_range, high_action_range)

    angles = get_current_motor_angles()

    if angles["GM"] + grab_action > -45 and angles["GM"] + grab_action < -230:
        grab_motor.run_angle(speed=motor_speed, rotation_angle=grab_action, wait=False)

    if angles["HM"] + high_action > 145 and angles["HM"] + high_action < -60:
        high_motor.run_angle(speed=motor_speed, rotation_angle=high_action, wait=False)

    if angles["LM"] + low_action > -10 and angles["LM"] + low_action < -190:
        low_motor.run_angle(speed=motor_speed, rotation_angle=low_action, wait=False)

    if angles["RM"] + rotation_action > 180 and angles["RM"] + rotation_action < -180:
        rotation_motor.run_angle(speed=motor_speed, rotation_angle=rotation_action, wait=False)

    wait(100)

    angles = get_current_motor_angles()
    out_msg = ustruct.pack('!ffff', normalize_angle(lf_angle), normalize_angle(rf_angle), normalize_angle(lb_angle), normalize_angle(rb_angle))
    stdout.buffer.write(out_msg)