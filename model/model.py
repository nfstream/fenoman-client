import tensorflow as tf
import flwr as fl
from patterns.singleton import singleton
from typing import Any
from data.data import data


@singleton
class Model:
    def __init__(self) -> None:
        # TODO majd ezt meg kell csinálni, hogy ha most lett compilolva akkor 1x tanítani kell, ha már meglévőt szeretnénk
        #   haszánlni akkor az nem szükséges az adatbázisból majd ezt is le kell tölteni!
        self.__model = tf.keras.Sequential()
        self.__model.add(tf.keras.layers.Dense(500, activation='relu'))
        self.__model.add(tf.keras.layers.Dense(100, activation='relu'))
        self.__model.add(tf.keras.layers.Dense(79))

        self.__model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

        x_train, y_train, x_val, y_val = data.load_data()
        print(len(x_train), len(y_train), len(x_val), len(y_val))
        history = self.__model.fit(
            x_train,
            y_train,
            batch_size=64,
            epochs=2,
            # We pass some validation for
            # monitoring validation loss and metrics
            # at the end of each epoch
            validation_data=(x_val, y_val),
        )

    def __call__(self) -> Any:
        return self.__model


model = Model()
