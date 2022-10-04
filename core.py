import flwr as fl
import requests

from pathlib import Path
from client.fenomanclient import FenomanClient
import argparse
from model.model import Model
from data.data import Data
from configuration.core_configuration import *


class Core:
    def __init__(self, data_uri: str, uri: str = URI, core_port: str = CORE_PORT, base_uri: str = BASE_URI) -> None:
        self.__uri = uri
        self.__core_port = core_port
        self.__base_uri = base_uri

        self.__data = Data()
        self.__data.replace_data(data_uri)
        self.__data.preprocess_data()

    def train(self, uri: str = URI, port: str = FENOMAN_CLIENT_PORT, secure: bool = SECURE_MODE) -> None:
        # TODO data set még szar meg kell oldani az is értelmes legyen
        self.__model.train()
        x_train, y_train, x_test, y_test = self.__data.load_data()

        # Start Flower client
        client = FenomanClient(self.__model, x_train, y_train, x_test, y_test)

        client_configuration = {
            'server_address': f'{uri}:{port}',
            'client': client
        }
        if secure:
            client_configuration['root_certificates'] = Path(".cache/certificates/ca.crt").read_bytes()
        fl.client.start_numpy_client(**client_configuration)

    def set_model(self, chosen_model: str) -> None:
        # TODO hibakezelés kell itt
        self.__model = Model(f'model/temp/{chosen_model}.h5')

    def __download_model(self, chosen_model: str) -> None:
        get_model_req = requests.get(f"{self.__base_uri}:{self.__core_port}{self.__base_uri}/get_model/{chosen_model}")
        open(f'model/temp/{chosen_model}.h5', 'wb').write(get_model_req.content)

    def predict(self, prediction_data):
        # TODO ezzel tud majd predictre használni majd, kell ide DATA Is persze :)
        classes = self.__model().predict_classes(prediction_data)
        return classes

    def get_models(self) -> list:
        get_models_req = requests.get(f'{self.__uri}:{self.__core_port}{self.__base_uri}/get_available_models')
        available_models = get_models_req.text.split(",")

        for x in range(len(available_models)):
            available_models[x] = available_models[x].split("\"")[1]

        return available_models


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Describe the csv number')
    parser.add_argument('-i', dest='i', help='CSV number label')
    args = parser.parse_args()
    new_data_uri = f'./data/comnet14-flows-part-{args.i}.csv'

    core = Core(new_data_uri)
    available_models = core.get_models()

    chosen_model = available_models[0]

    core.set_model(chosen_model)
    core.predict()

    core.train()
