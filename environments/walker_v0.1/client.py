# NOTE: Run this program with the latest
# firmware provided via https://beta.pybricks.com/

from pybricks.hubs import InventorHub
from pybricks.pupdevices import Motor, UltrasonicSensor, ColorSensor
from pybricks.parameters import Port, Direction
from pybricks.tools import wait
from pybricks.robotics import DriveBase
from pybricks.parameters import Axis

# Standard MicroPython modules
from usys import stdin, stdout
from uselect import poll
import ustruct
from micropython import kbd_intr
import umath

kbd_intr(-1)

hub = InventorHub()

# Initialize the drive base.
lf_motor = Motor(Port.D, Direction.COUNTERCLOCKWISE)
lb_motor = Motor(Port.B, Direction.COUNTERCLOCKWISE)
rf_motor = Motor(Port.C)
rb_motor = Motor(Port.A)

eyes = UltrasonicSensor(Port.E)

keyboard = poll()
keyboard.register(stdin)

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

(lf_angle, rf_angle) = (lf_motor.angle(), rf_motor.angle())
(lb_angle, rb_angle) = (lb_motor.angle(), rb_motor.angle())
low_angle = -150 # 270
high_angle = 150

while True:
    
    while not keyboard.poll(0):
        wait(1)

    # Read action values for both motors
    data = stdin.buffer.read(32)  # Reading 32 bytes (4 floats)

    lf_value, lb_value, rf_value, rb_value, lf_speed, lb_speed, rf_speed, rb_speed = ustruct.unpack("!ffffffff", data)

    lf_speed = transform_range(lf_speed, -1, 1, 200, 1000)
    lb_speed = transform_range(lb_speed, -1, 1, 200, 1000)
    rf_speed = transform_range(rf_speed, -1, 1, 200, 1000)
    rb_speed = transform_range(rb_speed, -1, 1, 200, 1000)

    # set motor angle. Range is [-180, 179] action outputs are [-1, 1]
    lb_motor.run_angle(speed=lb_speed, rotation_angle=transform_range(lb_value, -1, 1, low_angle, high_angle), wait=False)
    lf_motor.run_angle(speed=lf_speed,rotation_angle=transform_range(lf_value, -1, 1, low_angle, high_angle), wait=False)
    rb_motor.run_angle(speed=rb_speed,rotation_angle=transform_range(rb_value, -1, 1, low_angle, high_angle), wait=False)
    rf_motor.run_angle(speed=rf_speed,rotation_angle=transform_range(rf_value, -1, 1, low_angle, high_angle), wait=False) 

    a_x = hub.imu.acceleration(Axis.X)
    # Small delay to let motors arrive target angle
    wait(100) 
    # measure it before or after the wait? in between? two 200 wait times?
    

    # get current state of the robot
    (lf_angle, rf_angle) = (lf_motor.angle(), rf_motor.angle())
    (lb_angle, rb_angle) = (lb_motor.angle(), rb_motor.angle())
    (pitch, roll) = hub.imu.tilt()
    
    dist = eyes.distance()
    
    if umath.fabs(pitch) > 90 or umath.fabs(roll) > 120 or dist <= 40:
        hub.display.text(text="Help", on=500, off=50)

    # send current state
    # TODO: add current battery supply in mA for wandb logging
    #print("Battery: ", hub.battery.current())
    out_msg = ustruct.pack('!fffffff', normalize_angle(lf_angle), normalize_angle(rf_angle), normalize_angle(lb_angle), normalize_angle(rb_angle), pitch, roll, a_x)
    stdout.buffer.write(out_msg)

