import tensorflow as tf
import flwr as fl
from patterns.singleton import singleton
from typing import Any
from data.data import Data
from sklearn.preprocessing import LabelEncoder


@singleton
class Model:
    def __init__(self, model_config) -> None:
        self.__model = tf.keras.models.load_model(model_config)

    def train(self):
        lb = LabelEncoder()
        data = Data()
        x_train, y_train, x_val, y_val = data.load_data()

        y_val = lb.fit_transform(y_val)
        y_train = lb.fit_transform(y_train)

        print(len(x_train), len(y_train), len(x_val), len(y_val))
        history = self.__model.fit(
            x_train,
            y_train,
            batch_size=64,
            epochs=1,
            # We pass some validation for
            # monitoring validation loss and metrics
            # at the end of each epoch
            validation_data=(x_val, y_val),
        )

    def __call__(self) -> Any:
        return self.__model
