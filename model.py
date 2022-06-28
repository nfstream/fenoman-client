
class Model:

    def __init__(self):
        self.data = []

    def predict(self, flows):
        print("Anomaly or not")

    def load_data(self):
        flow_recor = open("flow_record.txt", "r")
        for flow in flow_recor:
            self.data.append(flow)
        flow_recor.close()

    def train(self):
        print("")

    def export_weights(self):
        print("")
