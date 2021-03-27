
from pybricks.pupdevices import Motor
from pybricks.parameters import Port
from pybricks.tools import wait
from urandom import randint

def _calibrate(motor,speed,duty=70):
    motor.run_until_stalled(speed,duty_limit=duty)
    return motor.angle()
    
    
class Robot:
    def __init__(self):
        motor_a = Motor(Port.A,gears=[8,24])
        motor_b = Motor(Port.B,gears=[12,60])
        motor_c = Motor(Port.C)
        motor_d = Motor(Port.D,gears=[12,60])
        self.motors = (motor_a,motor_b,motor_c,motor_d)
        self.speed = 100

    def calibrate(self):
        motor_a,motor_b,motor_c,motor_d = self.motors
        s = self.speed
    
        #arm
        M = _calibrate(motor_d,s,duty=100)
        box_d = (M-80,M)

        #Hook
        m = _calibrate(motor_c,-s)
        M = _calibrate(motor_c,s)
        motor_c.run_target(s,0.5*(M-m)+m) #open
        box_c=(m,M)
        motor_c
    
        #tip
        m = _calibrate(motor_a,-s)
        M = _calibrate(motor_a,s) # keep it here
        box_a=(m,M)
        
        #base
        motor_b.run_angle(100,180)
        box_b = (0,360)
        
        self.actionspace = (box_a,box_b,box_c,box_d)
        print("actionspace: {}".format(self.actionspace))
        
        
    def act(self,actions):
        for i,a in enumerate(actions):
            self.motors[i].run_target(self.speed,a,wait=False)
            #self.motors[i].track_target(a)
            wait(50)

        obs = []
        for i,m in enumerate(self.motors):
            obs.append(m.angle())
        
        return obs
            

print("START")
    
robot = Robot()
robot.calibrate()
actions = [0.0,0.0,0.0,0.0]
for i in range(100):
    for ai in range(4):
        [m,M] = robot.actionspace[ai]
        actions[ai] = randint(m,M)
    obs = robot.act(actions)
    print('actions {}   obs: {}'.format(actions, obs))

wait(1000)
print("STOP")
