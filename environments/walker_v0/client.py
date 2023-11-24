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

#color = ColorSensor(Port.F)
eyes = UltrasonicSensor(Port.E)

# target angle: 179 is good starting position!
keyboard = poll()
keyboard.register(stdin)


while True:
    
    while not keyboard.poll(0):
        wait(1)

    # Read action values for both motors
    data = stdin.buffer.read(16)  # Reading 16 bytes (4 floats)

    lf_value, lb_value, rf_value, rb_value = ustruct.unpack("!ffff", data)

    # set motor angle. Range is [-180, 179] action outputs are [-1, 1] -> we multiply by 179
    lf_motor.track_target(target_angle=lf_value*179)
    lb_motor.track_target(target_angle=lb_value*179)
    rf_motor.track_target(target_angle=rf_value*179)
    rb_motor.track_target(target_angle=rb_value*179)

    wait(50)  # Small delay 

    # get current state of the robot
    (lf_angle, rf_angle) = (lf_motor.angle(), rf_motor.angle())
    (lb_angle, rb_angle) = (lb_motor.angle(), rb_motor.angle())
    (pitch, roll) = hub.imu.tilt()
    
    dist = eyes.distance()
    a_x = hub.imu.acceleration(Axis.X)

    # roll & pitch should be <50 and >-50
    if umath.fabs(pitch) > 50 or umath.fabs(roll) > 50 or dist <= 40:
        hub.display.text(text="Help", on=500, off=50)
        hub.speaker.beep(frequency=500, duration=100)

    # send current state
    out_msg = ustruct.pack('!fffffff', lf_angle, rf_angle, lb_angle, rb_angle, pitch, roll, a_x)
    stdout.buffer.write(out_msg)

