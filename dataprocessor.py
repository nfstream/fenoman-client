from nfstream import NFStreamer
import asyncio

class DataProcessor:
    def __init__(self, interface):
        self.interface = interface

    async def process(self):
        if self.interface == "":
            self.interface = "eth0"
            #Get default interface and replace eth0

        streamer = NFStreamer(source=self.interface)
        for flow in streamer:
            return flow

