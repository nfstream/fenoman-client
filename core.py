import flwr as fl
import requests
import json
import pandas as pd
import numpy as np
from typing import Any, Tuple, Union
from pathlib import Path
from client.fenomanclient import FenomanClient
from model.model import Model
from data.data import Data, data
from configuration.core_configuration import *
from configuration.data_configuration import *
from helpers.request_handler import request_handler


class Core:
    def __init__(self,
                 data_uri: str = DATA_URI,
                 server_protocol: str = SERVER_PROTOCOL,
                 server_uri: str = URI,
                 core_port: str = CORE_PORT,
                 base_uri: str = BASE_URI,
                 ocm_apim_key: str = OCM_APIM_KEY) -> None:
        """
        Initialization function for FeNOMan client instantiation.

        In order to use the FeNOMan client we need to instantiate it with core.Core(). You can of course pass in the CSV
        data uri from which the solution will work, but this can also be left empty. The available models are retrieved
        with get_models(), and then the selected model (chosen from the list returned) can be set with set_model().

        ```\n
        core = core.Core(new_data_uri)\n
        _, available_models = core.get_models()\n
        chosen_model = available_models[0]\n
        _, _ = core.set_model(chosen_model)\n
        ```

        In case you want to make predictions locally, predictions are generated by calling predict() on the pandas
        DataFrame after the set_model() procedure.

        ```\n
        prediction_data = pd.DataFrame()\n
        core.predict(prediction_data)\n
        ```

        If you want to do federated learning, you can do it with .train().

        ```\n
        core.train()\n
        ```

        :param data_uri: path of the data on the local environment
        :param server_protocol: HTTP or HTTPS protocol of the FeNOMan server
        :param server_uri: URI of the FeNOMan server
        :param core_port: port of the FeNOMan application server
        :param base_uri: application version base uri like /api/v1
        :param ocm_apim_key: application key to access server resources
        :return: None
        """
        self.__server_protocol = server_protocol
        self.__server_uri = server_uri
        self.__core_port = core_port
        self.__base_uri = base_uri

        self.__data = Data(data_uri)
        self.__data.preprocess_data()

        self.__port_mapping = None
        self.__fenoman_client_port = None

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
            f'{self.__server_protocol}://{self.__server_uri}:{self.__core_port}{self.__base_uri}/get_model/{chosen_model}',
            headers=self.__http_headers)
        request_state, request_content = request_handler.process_response(get_model_req)
        if request_state:
            open(f'model/temp/{chosen_model}.h5', 'wb').write(request_content)
            return True, ""
        else:
            return False, request_content.decode("utf-8")

    def train(self, uri: str = URI, secure: bool = SECURE_MODE) -> Tuple[bool, str]:
        """
        Using the train() method, we can train the set model along the federated learning paradigm. In the configuration
        it is important to specify the port number of the flower server and not the port of the application server.

        :param uri: base URI of the FeNOMan server
        :param port: port of the Flower server inside the FeNOMan server application
        :param secure: security mode turn on for SSL tunnel
        :return: None
        """
        self.__model.train()
        x_train, y_train, x_test, y_test = self.__data.load_data()

        # Start Flower client
        client = FenomanClient(self.__model, x_train, y_train, x_test, y_test)

        if self.__fenoman_client_port is None:
            return False, "The set_model() method should be called first!"

        client_configuration = {
            'server_address': f'{uri}:{self.__fenoman_client_port}',
            'client': client
        }

        if secure:
            client_configuration['root_certificates'] = Path('.cache/certificates/ca.crt').read_bytes()
        fl.client.start_numpy_client(**client_configuration)

        return True, "success"

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

            if self.__port_mapping is None:
                return False, "The get_models() method should be called first!"

            id_of_model = self.__port_mapping['models'].index(chosen_model)
            self.__fenoman_client_port = self.__port_mapping['ports'][id_of_model]

            return True, "success"
        else:
            return False, download_resp

    def predict(self, prediction_data: str) -> Any:
        """
        In case you want to request a prediction, you can use this function after you have instantiated the FeNOMan
        Core and set the desired model with set_model(). The set_model() expects the model name which can be retrieved
        with get_models() if we don't know the available models.

        :param prediction_data: input for which the predictions should be made
        :return: labeled classes
        """
        data.replace_data(prediction_data)
        data.preprocess_data()
        prediction_data_preprocessed_part1,_,prediction_data_preprocessed_part2,_ = data.load_data()
        prediction_data_preprocessed = pd.concat(
            [prediction_data_preprocessed_part1, prediction_data_preprocessed_part2]
        )

        classes = self.__model().predict(prediction_data_preprocessed)
        return classes

    def get_models(self) -> Tuple[bool, Union[list, str]]:
        """
        Returns the list of models available on the server. Each element in the list is a string and should be used to
        refer to the set_model() method to determine which model to use.

        :return: state and the list of responses
        """
        get_models_req = requests.get(f'{self.__server_protocol}://{self.__server_uri}:{self.__core_port}{self.__base_uri}/get_available_models',
                                      headers=self.__http_headers)
        request_state, request_content = request_handler.process_response(get_models_req)
        if request_state:
            self.__port_mapping = json.loads(request_content.decode("utf-8"))
            available_models = self.__port_mapping['models']
            return True, available_models
        else:
            return False, request_content.decode("utf-8")

    @staticmethod
    def get_decode_classes(predictions) -> list:
        """
        This function gives back the actual class names based on the dataset target field.
        TODO future development required in order to eliminate the dataset target field extraction.

        :param predictions: prediction matrices from the core.predict() function.
        :return: list of decoded classes
        """
        field_names = data.get_target_names()
        prediction_fields = [field_names[np.argmax(i)] for i in predictions]
        return prediction_fields
