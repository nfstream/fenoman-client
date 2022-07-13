from data_processor.data_processor import data_processor
from model.model import nfstream_client
import flwr as fl
import pathlib


def main():
    (x_train, y_train), (x_test, y_test) = data_processor.load_partition(...)

    # Start Flower client
    client = nfstream_client(model, x_train, y_train, x_test, y_test)

    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=client,
        root_certificates=pathlib.Path("certificates/ca.crt").read_bytes(),
    )


if __name__ == '__main__':
    main()
