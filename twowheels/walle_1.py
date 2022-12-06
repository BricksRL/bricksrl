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

    # Read three bytes.
    cmd = stdin.buffer.read(1)

    # Decide what to do based on the command.
    if cmd == b"a":#forward
        drive_base.straight(200)
    elif cmd == b"z":#back
        drive_base.straight(-200)
    elif cmd == b"k":#left
        drive_base.turn(-45)
    elif cmd == b"l":#right
        drive_base.turn(45)
    elif cmd == b"p":
        hub.speaker.play_notes(["A7/16", "G7/16", "E7/16", "C7/16","D7/16","B7/16","F7/16","C8/16","A7/16","G7/16","E7/16"])
    elif cmd == b"b":
        break
    else:
        drive_base.stop()
    
    # Send a response.
    (left, right) = (left_motor.angle(), right_motor.angle())
    (pitch, roll) = hub.imu.tilt()
    dist = sensor.distance()
    
    buffer = ustruct.pack("ddddd",left,right,pitch,roll,dist)
    stdout.buffer.write(buffer)
#    stdout.buffer.write(b"OBS: [{}, {}, {}, {}, {}] ".format(left,right,pitch,roll,dist))

