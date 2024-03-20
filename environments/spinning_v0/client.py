import ustruct
from micropython import kbd_intr
from pybricks.hubs import InventorHub
from pybricks.parameters import Axis, Direction, Port
from pybricks.pupdevices import Motor, UltrasonicSensor
from pybricks.robotics import DriveBase
from pybricks.tools import wait
from uselect import poll

# Standard MicroPython modules
from usys import stdin, stdout

kbd_intr(-1)

hub = InventorHub()

# Initialize the drive base.
left_motor = Motor(Port.E, Direction.COUNTERCLOCKWISE)
right_motor = Motor(Port.A)
# drive_base = DriveBase(left_motor, right_motor, wheel_diameter=56, axle_track=130)
# Initialize the distance sensor.

# Optional: Register stdin for polling. This allows
# you to wait for incoming data without blocking.
keyboard = poll()
keyboard.register(stdin)


def normalize_angle(angle):
    # Normalize angle to be within 0 and 360
    while angle <= 0:
        angle += 360
    while angle > 360:
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


while True:

    # Optional: Check available input.
    while not keyboard.poll(0):
        wait(1)

    # Read action values for both motors
    data = stdin.buffer.read(8)  # Reading 8 bytes (two floats)
    left_action_value, right_action_value = ustruct.unpack("!ff", data)

    # Apply action to each motor
    left_motor.run_angle(
        speed=400,
        rotation_angle=transform_range(left_action_value, -1, 1, -100, 100),
        wait=False,
    )
    right_motor.run_angle(
        speed=400,
        rotation_angle=transform_range(right_action_value, -1, 1, -100, 100),
        wait=False,
    )

    wait(100)  # Small delay

    # get current state of the robot
    (left, right) = (left_motor.angle(), right_motor.angle())
    (pitch, roll) = hub.imu.tilt()
    z_angl_vel = hub.imu.angular_velocity(Axis.Z)

    # send current state
    out_msg = ustruct.pack(
        "!fffff", normalize_angle(left), normalize_angle(right), pitch, roll, z_angl_vel
    )
    stdout.buffer.write(out_msg)
