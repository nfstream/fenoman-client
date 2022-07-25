from patterns.singleton import singleton
import flwr as fl
import model.decision_tree_model

@singleton
class NFStreamClient(fl.client.NumPyClient):
    def __init__(self, model, train_data, val_data):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data

    def get_properties(self, config):
        """Get properties of client."""
        raise Exception("Not implemented")

    def get_parameters(self):
        """Get parameters of the local model."""
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        decision_tree_model.train(self.model, self.train_data)
        return self.model.get_weights(), len(self.train_data), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = decision_tree_model.test(self.model, self.val_data)
        print("Eval accuracy : ", accuracy)

        #@TODO loss is not computed correctly
        loss = 1 - accuracy
        print("loss : ", loss)
        return loss, len(self.train_data), {"accuracy": accuracy}


nfstream_client = NFStreamClient()
