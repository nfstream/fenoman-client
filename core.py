import flwr as fl
import requests
import pandas as pd
from typing import Any, Tuple, Union
from pathlib import Path
from client.fenomanclient import FenomanClient
from model.model import Model
from data.data import Data
from configuration.core_configuration import *
from helpers.request_handler import request_handler


class Core:
    def __init__(self,
                 data_uri: str,
                 server_uri: str = URI,
                 core_port: str = CORE_PORT,
                 base_uri: str = BASE_URI,
                 ocm_apim_key: str = OCM_APIM_KEY) -> None:
        """
        Initialization function for FeNOMan client instantiation.

        :param data_uri: path of the data on the local environment
        :param uri: URI of the FeNOMan server
        :param core_port: port of the FeNOMan application server
        :param base_uri: application version base uri like /api/v1
        :param ocm_apim_key: application key to access server resources
        :return: None
        """
        self.__server_uri = server_uri
        self.__core_port = core_port
        self.__base_uri = base_uri

        # TODO data nem lesz igy jÓ
        self.__data = Data()
        self.__data.replace_data(data_uri)
        self.__data.preprocess_data()

        self.__http_headers = {
            'Ocp-Apim-Key': ocm_apim_key
        }

    def __download_model(self, chosen_model: str) -> Tuple[bool, str]:
        """
        Internal function to download the model from the server. The model is stored at file level temporarily on the
        computer not in memory at runtime.

        :param chosen_model: name of the model that must be downloaded from the server
        :return: state of the request success and the response as bytes or str
        """
        get_model_req = requests.get(
            f'{self.__server_uri}:{self.__core_port}{self.__base_uri}/get_model/{chosen_model}',
            headers=self.__http_headers)
        request_state, request_content = request_handler.process_response(get_model_req)
        if request_state:
            open(f'model/temp/{chosen_model}.h5', 'wb').write(request_content)
            return True, ""
        else:
            return False, request_content.decode("utf-8")

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
            client_configuration['root_certificates'] = Path('.cache/certificates/ca.crt').read_bytes()
        fl.client.start_numpy_client(**client_configuration)

    def set_model(self, chosen_model: str) -> Tuple[bool, str]:
        """
        This method tells the Core which model to use, and retrieves the appropriate model data from the FeNOMan
        server.

        :param chosen_model: name of the model that must be downloaded from the server
        :return: None
        """
        download_state, download_resp = self.__download_model(chosen_model)
        if download_state:
            self.__model = Model(f'model/temp/{chosen_model}.h5')
            return True, "success"
        else:
            return False, download_resp

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

    def get_models(self) -> Tuple[bool, Union[list, str]]:
        """
        Returns the list of models available on the server. Each element in the list is a string and should be used to
        refer to the set_model() method to determine which model to use.

        :return: state and the list of responses
        """
        get_models_req = requests.get(f'{self.__server_uri}:{self.__core_port}{self.__base_uri}/get_available_models',
                                      headers=self.__http_headers)
        request_state, request_content = request_handler.process_response(get_models_req)
        if request_state:
            available_models = get_models_req.text.split(",")
            return True, available_models
        else:
            return False, request_content.decode("utf-8")

