import flwr as fl
from sklearn.preprocessing import LabelEncoder
from model.model import Model
from typing import Any, Union
from configuration.client_configuration import *


class FenomanClient(fl.client.NumPyClient):
    def __init__(self, model: Model, x_train: Any, y_train: Any, x_test: Any, y_test: Any) -> None:
        """
        A NumPyClient based NFStream compliant instantiation of the flower component can be done with this to make it
        compatible with FeNOMan.

        :param model: input model that is downloaded from the FeNOMan server
        :param x_train: Teaching dataset containing the features.
        :param y_train: Target variable for teaching.
        :param x_test: Testing dataset containing the features.
        :param y_test: Target variable for testing.
        :return: None
        """
        self.model = model()
        lb = LabelEncoder()

        self.y_test = lb.fit_transform(y_test)
        self.y_train = lb.fit_transform(y_train)
        self.x_train = x_train
        self.x_test = x_test

    def get_properties(self, config: dict) -> Exception:
        """
        Return properties of the client. Currently, unsupported method.

        :param config: Configuration parameters which allow the server to influence training on the client. It can be
        used to communicate arbitrary values from the server to the client, for example, to set the number of (local)
        training epochs.
        :return: Exception
        """
        raise Exception("Not implemented")

    def get_parameters(self) -> Exception:
        """
        Returns the parameter of the local model. Currently, unsupported method.

        :return: Exception
        """
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config) -> Union[Any, int, dict]:
        """
        This function train parameters on the locally held training set.

        :param parameters: model parameters
        :param config: configuration of train method
        :return: updated model parameters, train count and results
        """
        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyper parameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        # Train the model using hyper parameters from config
        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size,
            epochs,
            validation_split=VALIDATION_SPLIT,
        )

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
        }
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters: Any, config: dict) -> Union[Any, int, dict]:
        """
        Evaluate parameters on the locally held test set.

        :param parameters: model parameters
        :param config: model configuration
        :return: model parameters on the local test data and return results
        """
        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, EVALUATE_BATCH_SIZE, steps=steps)
        num_examples_test = len(self.x_test)
        return loss, num_examples_test, {"accuracy": accuracy}
