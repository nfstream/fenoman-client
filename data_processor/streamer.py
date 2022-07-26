import nfstream
import model.decision_tree_model

class Streamer:
    def __init__(self, interface):
        self.streamer = nfstream.NFStreamer(source=interface, idle_timeout=20)
        self.flows = []
        self.predictions = []

    def run(self):
        for flow in self.streamer:
            prediction = model.decision_tree_model.predict(flow)
            self.predictions.append(prediction)
            self.flows.append(flow)
            with open("flow_record.txt", "a") as f:
                f.write(str(flow))
