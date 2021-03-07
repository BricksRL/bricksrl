from pybricks.pupdevices import Motor
from pybricks.parameters import Port
from pybricks.tools import wait

#motor_a = Motor(Port.A)
#motor_a.run_until_stalled(20)


#motor_c = Motor(Port.C)
#motor_c.run_until_stalled(-10)
#print("Stalled")

motor_b = Motor(Port.B)
motor_b.run(-200)

motor_d = Motor(Port.D)
motor_d.run_until_stalled(-50)

wait(5000)
