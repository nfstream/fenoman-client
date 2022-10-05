import tensorflow as tf
from patterns.singleton import singleton
from typing import Any
from data.data import Data
from sklearn.preprocessing import LabelEncoder
from configuration.client_configuration import *


@singleton
class Model:
    def __init__(self, model_config: str) -> None:
        """
        This is the base class of the Keras Model which is created by a model.h5 configuration that must be downloaded
        from the FeNOMan server.

        :param model_config: url of the model.h5 file
        :return: None
        """
        self.__model = tf.keras.models.load_model(model_config)

    def train(self) -> None:
        """
        Train method that fits the data ...

        :return: None
        """
        # TODO nem jó mert ez itt nem független a szervertől lol
        lb = LabelEncoder()
        data = Data()
        x_train, y_train, x_val, y_val = data.load_data()

        y_val = lb.fit_transform(y_val)
        y_train = lb.fit_transform(y_train)

        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        self.__history = self.__model.fit(
            x_train,
            y_train,
            batch_size=TRAIN_BATCH_SIZE,
            epochs=TRAIN_EPOCHS,
            validation_data=(x_val, y_val),
        )

    def __call__(self) -> Any:
        """
        This __call__ method used to call directly the underlaying keras model. This allows the developer to use the
        built-in keras functions if the Model object is called like model().

        :return: Keras Model
        """
        return self.__model
