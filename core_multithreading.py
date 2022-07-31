import flwr as fl
from model.model import model
from client.cifarclient import CifarClient
from data.data import data
from queue import Queue
from threading import Thread


NUM_THREADS = 5
q = Queue()


def run_logic():
    global q

    file_uri = q.get()
    data.replace_data(file_uri)
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
    q.task_done()


def main():
    global q

    for i in range(1, 6, 1):
        q.put(f'./data/comnet14-flows-part-{i}.csv')

    for t in range(NUM_THREADS):
        worker = Thread(target=run_logic)
        worker.daemon = True
        worker.start()

    q.join()


if __name__ == '__main__':
    main()
