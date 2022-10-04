import tensorflow as tf
from patterns.singleton import singleton
from typing import Any
from data.data import Data
from sklearn.preprocessing import LabelEncoder


@singleton
class Model:
    def __init__(self, model_config) -> None:
        # TODO
        self.__model = tf.keras.models.load_model(model_config)

    def train(self) -> None:
        # TODO
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
            batch_size=64,
            epochs=1,
            validation_data=(x_val, y_val),
        )

    def __call__(self) -> Any:
        # TODO
        return self.__model
