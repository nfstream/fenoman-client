import flwr as fl
from model.model import model
from client.cifarclient import CifarClient
from data.data import data


def main():
    x_train, y_train, x_test, y_test = data.load_data()

    # Start Flower client
    client = CifarClient(model, x_train, y_train, x_test, y_test)

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=client,
        #root_certificates=Path(".cache/certificates/ca.crt").read_bytes(),
    )


if __name__ == '__main__':
    main()