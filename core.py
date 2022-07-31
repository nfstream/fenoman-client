import flwr as fl
from client.cifarclient import CifarClient
import argparse
from model.model import model
from data.data import data
from configuration import data_configuration


def main():
    data.preprocess_data()
    model.train()
    x_train, y_train, x_test, y_test = data.load_data()

    # Start Flower client
    client = CifarClient(model, x_train, y_train, x_test, y_test)

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=client,
        #root_certificates=Path(".cache/certificates/ca.crt").read_bytes(),
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Describe the csv number')
    parser.add_argument('-i', dest='i', help='CSV number label')
    args = parser.parse_args()
    new_data_uri = f'./data/comnet14-flows-part-{args.i}.csv'
    data.replace_data(new_data_uri)
    main()
