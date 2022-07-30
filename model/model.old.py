from patterns.singleton import singleton
import tensorflow as tf
from typing import Any


@singleton
class Model:
    def __init__(self) -> None:
        self.__model = tf.keras.applications.EfficientNetB0(
            input_shape=(32, 32, 3), weights=None, classes=10
        )
        self.__model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    def __call__(self) -> Any:
        return self.__model


model = Model()
