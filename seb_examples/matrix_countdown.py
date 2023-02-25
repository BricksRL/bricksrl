from pybricks.hubs import PrimeHub
from pybricks.tools import wait

# init hub
hub = PrimeHub()

for i in reversed(range(5)):
    hub.display.number(i)
    wait(500)