import flwr as fl
import requests
import pandas as pd

from typing import Any
from pathlib import Path
from client.fenomanclient import FenomanClient
import argparse
from model.model import Model
from data.data import Data
from configuration.core_configuration import *


class Core:
    def __init__(self, data_uri: str, uri: str = URI, core_port: str = CORE_PORT, base_uri: str = BASE_URI) -> None:
        """
        Initialization function for FeNOMan client instantiation.

        :param data_uri: path of the data on the local environment
        :param uri: URI of the FeNOMan server
        :param core_port: port of the FeNOMan application server
        :param base_uri: application version base uri like /api/v1
        :return: None
        """
        self.__uri = uri
        self.__core_port = core_port
        self.__base_uri = base_uri

        # TODO data nem lesz igy jÓ
        self.__data = Data()
        self.__data.replace_data(data_uri)
        self.__data.preprocess_data()

    def __download_model(self, chosen_model: str) -> None:
        """
        Internal function to download the model from the server. The model is stored at file level temporarily on the
        computer not in memory at runtime.

        :param chosen_model: name of the model that must be downloaded from the server
        :return: None
        """
        get_model_req = requests.get(f"{self.__base_uri}:{self.__core_port}{self.__base_uri}/get_model/{chosen_model}")
        open(f'model/temp/{chosen_model}.h5', 'wb').write(get_model_req.content)

    def train(self, uri: str = URI, port: str = FENOMAN_CLIENT_PORT, secure: bool = SECURE_MODE) -> None:
        """
        Using the train() method, we can train the set model along the federated learning paradigm. In the configuration
        it is important to specify the port number of the flower server and not the port of the application server.

        :param uri: base URI of the FeNOMan server
        :param port: port of the Flower server inside the FeNOMan server application
        :param secure: security mode turn on for SSL tunnel
        :return: None
        """
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
        """
        This method tells the Core which model to use, and retrieves the appropriate model data from the FeNOMan
        server.

        :param chosen_model: name of the model that must be downloaded from the server
        :return: None
        """
        self.__download_model(chosen_model)
        self.__model = Model(f'model/temp/{chosen_model}.h5')

    def predict(self, prediction_data: pd.DataFrame) -> Any:
        """
        In case you want to request a prediction, you can use this function after you have instantiated the FeNOMan
        Core and set the desired model with set_model(). The set_model() expects the model name which can be retrieved
        with get_models() if we don't know the available models.

        :param prediction_data: input for which the predictions should be made
        :return: labeled classes
        """
        classes = self.__model().predict_classes(prediction_data)
        return classes

    def get_models(self) -> list:
        """
        Returns the list of models available on the server. Each element in the list is a string and should be used to
        refer to the set_model() method to determine which model to use.

        :return: list
        """
        get_models_req = requests.get(f'{self.__uri}:{self.__core_port}{self.__base_uri}/get_available_models')
        available_models = get_models_req.text.split(",")

        for x in range(len(available_models)):
            available_models[x] = available_models[x].split("\"")[1]

        return available_models


if __name__ == '__main__':
    # TODO ezeket majd ki kell törölni nem kellenek
    parser = argparse.ArgumentParser(description='Describe the csv number')
    parser.add_argument('-i', dest='i', help='CSV number label')
    args = parser.parse_args()
    new_data_uri = f'./data/comnet14-flows-part-{args.i}.csv'

    core = Core(new_data_uri)
    available_models = core.get_models()

    chosen_model = available_models[0]

    core.set_model(chosen_model)

    prediction_data = pd.DataFrame()
    core.predict(prediction_data)

    core.train()
