from dataprocessor import DataProcessor

class Connector:

    def __init__(self, interface):
        self.running = False
        self.interface = interface
        self.flows = []

    def subscribe(self):
        #Communication with the server

        #Start the flow
        self.running = True
        dp = DataProcessor(interface=self.interface)

        while self.running:
            flow = dp.process()
            flow_record = open("flow_record.txt", "a")
            flow_record.write(flow)
            flow_record.close()
            self.flows.append(flow)

    def unsubscribe(self):
        #Stopping the loop
        self.running = False

        #Communication with the server
