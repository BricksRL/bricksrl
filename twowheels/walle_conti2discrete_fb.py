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
    # All forward commands 
    if cmd == b"a": # forward 
        drive_base.straight(100)
    elif cmd == b"b": # forward
        drive_base.straight(90)
    elif cmd == b"c": # forward
        drive_base.straight(80)
    elif cmd == b"d": # forward
        drive_base.straight(70)
    elif cmd == b"e": # forward
        drive_base.straight(60)
    elif cmd == b"f": # forward
        drive_base.straight(50)
    elif cmd == b"g": # forward
        drive_base.straight(40)
    elif cmd == b"h": # forward
        drive_base.straight(30)
    elif cmd == b"i": # forward
        drive_base.straight(20)
    elif cmd == b"j": # forward
        drive_base.straight(10)
    elif cmd == b"k": # forward
        drive_base.straight(0)
          
    # All backward commands
    elif cmd == b"u": # forward 
        drive_base.straight(-100)
    elif cmd == b"t": # forward
        drive_base.straight(-90)
    elif cmd == b"s": # forward
        drive_base.straight(-80)
    elif cmd == b"r": # forward
        drive_base.straight(-70)
    elif cmd == b"q": # forward
        drive_base.straight(-60)
    elif cmd == b"p": # forward
        drive_base.straight(-50)
    elif cmd == b"o": # forward
        drive_base.straight(-40)
    elif cmd == b"n": # forward
        drive_base.straight(-30)
    elif cmd == b"m": # forward
        drive_base.straight(-20)
    elif cmd == b"l": # forward
        drive_base.straight(-10)
 
    # quit
    elif cmd == b"y":
        break
    else:
        drive_base.stop()
    
    # Send a response.
    (left, right) = (left_motor.angle(), right_motor.angle())
    (pitch, roll) = hub.imu.tilt()
    dist = sensor.distance()
    
    buffer = ustruct.pack('!iiiii',left,right,pitch,roll,dist)
    stdout.buffer.write(buffer)
#    stdout.buffer.write(b"OBS: [{}, {}, {}, {}, {}] ".format(left,right,pitch,roll,dist))

