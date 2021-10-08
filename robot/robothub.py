
from pybricks.pupdevices import Motor
from pybricks.parameters import Port
from pybricks.tools import wait
from urandom import randint
from uselect import poll
from usys import stdin
import ustruct

def _calibrate(motor,speed,duty=70):
    motor.run_until_stalled(speed,duty_limit=duty)
    return motor.angle()
    

def update_input(char,input_buffer):
    cmd = None
    if char == "$":
        cmd = input_buffer
        input_buffer = ""
    else:
        input_buffer += char
    return cmd,input_buffer
    
    
class RobotHub:
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
        
    def _get_obs(self):
        obs = []
        for i,m in enumerate(self.motors):
            obs.append(m.angle())
        return obs

    def act(self,actions):
        for i,a in enumerate(actions):
            self.motors[i].run_target(self.speed,a,wait=False)
            #self.motors[i].track_target(a)
            wait(50)

        obs = self._get_obs()
        return obs
            
    def main_loop(self):
        input_buffer = ""
        loop_poll = poll()
        loop_poll.register(stdin)
        running = True
        while running:
            if loop_poll.poll(100): #times out after 100ms
                char = stdin.read(1)
                if char is not None:
                    cmd, input_buffer = update_input(char)
                if cmd is not None:
                    actions = ustruct.unpack('4f',cmd)
                    obs = self.act(actions)

                    print(obs)

            # update other stuff here

    def reset(self):
        pass

print("START")
running = True
    
robot = RobotHub()
robot.calibrate()
robot.main_loop()

wait(1000)
print("STOP")
