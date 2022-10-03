import flwr as fl
import requests

from client.cifarclient import CifarClient
import argparse
from model.model import Model
from data.data import Data
from configuration import data_configuration


class Core:
    def __init__(self, data_uri):
        self.data = Data()
        self.data.replace_data(data_uri)
        self.data.preprocess_data()

        get_models_req = requests.get("http://127.0.0.1:5000/api/v1/get_avilable_models")
        avilable_models = get_models_req.text.split(",")

        for x in range(len(avilable_models)):
            avilable_models[x] = avilable_models[x].split("\"")[1]
            print(str(x) + ".: " + avilable_models[x])

        chosen_model = int(input("Choose a model: "))
        get_model_req = requests.get("http://127.0.0.1:5000/api/v1/get_model/" + avilable_models[chosen_model])
        open("model/temp/" + avilable_models[chosen_model] + ".h5", "wb").write(get_model_req.content)
        self.model = Model("model/temp/" + avilable_models[chosen_model] + ".h5")

        self.run_fenoman_client()

    def run_fenoman_client(self, server_address: str = "127.0.0.1:8080"):
        # TODO: download all the configuration from the server

        self.model.train()
        x_train, y_train, x_test, y_test = self.data.load_data()

        # Start Flower client
        client = CifarClient(self.model, x_train, y_train, x_test, y_test)

        fl.client.start_numpy_client(
            server_address=server_address,
            client=client,
            #root_certificates=Path(".cache/certificates/ca.crt").read_bytes(),
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Describe the csv number')
    parser.add_argument('-i', dest='i', help='CSV number label')
    args = parser.parse_args()
    new_data_uri = f'./data/comnet14-flows-part-{args.i}.csv'
    core = Core(new_data_uri)
