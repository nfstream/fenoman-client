from connector import Connector
from model import Model
import asyncio

class Core:

    def __init__(self, ip = "", port = 9000, interface = ""):
        self.server_ip = ip
        self.server_port = port
        self.interface = interface

    def run(self):
        self.create_connection()

        connector = Connector(interface=self.interface)
        asyncio.run(connector.subscribe())

        model = Model()

        #Implement ctrl + c to turn off app here
        while True:
            if len(connector.flows) > 10:
                model.predict(connector.flows)
                connector.flows.clear()

    def create_connection(self):
        print("Creation connection with " + self.server_ip + ":" + str(self.server_port))
        #Rest API querrys come here

    def load_model(self):
        print("")

core = Core("localhost",9000,"\\Device\\NPF_{5C19BF9B-2A9D-45C2-B292-07C140E6EF3F}")
core.run()
