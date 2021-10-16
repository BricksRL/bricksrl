import asyncio
from mmap import ACCESS_DEFAULT

class Test:
    def __init__(self, sleep) -> None:
        self.sleep = sleep
        print("Start",flush=True)
        asyncio.run(self.main())
        print("Stop",flush=True)
        
    async def main(self):
        await asyncio.sleep(self.sleep)
        run_task = asyncio.create_task(self.hub_run())
        while True:
            await asyncio.sleep(1)


    async def hub_run(self):
        print("hub_run",flush=True)
        for i in range(1000):
            print("HUB: Hello {}".format(i),flush=True)
            await asyncio.sleep(0.1)
        return 0

test = Test(4)
