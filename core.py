from os.path import exists

import numpy as np
import pandas as pd

#from data_processor.data_processor import data_processor
import model.decision_tree_model
#from model.model import nfstream_client
#import flwr as fl
#import pathlib
import requests
import json
import keras
import model.headers

def init_connection():
    identifier = 0
    if exists("configuration/identifier.txt"):
        with open("configuration/identifier.txt") as f:
            identifier = int(f.read())
    else:
        req = requests.get("http://147.232.207.111:80/register")
        identifier = int(req.text)
        with open("configuration/identifier.txt", "w") as f:
            f.write(req.text)

    return identifier


def download_file(url, fname):
    local_filename = url.split('/')[-1]
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(fname, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                # If you have chunk encoded response uncomment if# and set chunk_size parameter to None.#if chunk:
                f.write(chunk)


def load_server_model():
    get_model_endpoint = "http://147.232.207.111:80/get/model"
    get_weights_endpoint = "http://147.232.207.111:80/get/weights"

    download_file(get_model_endpoint, "model.json")  # Downloading the newest model from the server
    download_file(get_weights_endpoint, "weights.h5")  # Downloading the model weights from the server

    # Loading model from a file as a json
    f = open('model.json')
    server_model = json.load(f)
    f.close()

    # Load model from json
    raw_model = keras.models.model_from_json(server_model, custom_objects={'NeuralDecisionTree': model.decision_tree_model.NeuralDecisionTree})
    raw_model.load_weights("weights.h5")

    return raw_model


def push_model(trained_model):
    model_json = trained_model.to_json()
    post_endpoint = "http://147.232.207.111:80/send/model"
    r = requests.post(post_endpoint, json=model_json)
    print("Posting the model: " + str(r.status_code))


def push_weights(trained_model):
    trained_model.save_weights("weights.h5")
    files = {'file': ("weights.h5", open('weights.h5', 'rb'))}
    weight_endpoint = "http://147.232.207.111:80/send/weights"
    r2 = requests.post(weight_endpoint, files=files)
    print("Posting the weights: " + str(r2.status_code))


def main():
    identifier = init_connection()

    data = [] #Should be loaded from Nfstream generated csv
    treshold = 0 #The minimum size of the csv, that is sufficient to train the model
    if(len(data) >= treshold):
        print("Ping the server that we are ready to train the model")

    #Transfer learning with own data
    """Simon = server"""
    SimonSaysYes = True
    if (SimonSaysYes):
        raw_model = load_server_model()
        print("Server model loaded")
        #To be deleted
        test_data = model.headers.test_data
        #model.decision_tree_model.test(raw_model, test_data)

        print("Transfer training:")
        train_data = model.headers.transfer_data
        trained_model = model.decision_tree_model.train(raw_model, train_data)

        print("Testing trained model:")
        #To be deleted
        model.decision_tree_model.test(trained_model, test_data)

        #Pushing new model to the server
        push_model(trained_model)

        #Pushing the model weights to the server
        push_weights(trained_model)









    #----------------------------------------------------------------
    '''csv = pd.read_csv("comnet14-flows.csv")
    csv = csv.iloc[:, :-5]
    df_split = np.array_split(csv, 8)
    train_data = df_split[3]
    test_data = df_split[4]

    model = decision_tree_model.create_tree_model()
    # Start Flower client
    client = nfstream_client(model, train_data, test_data)

    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=client,
        root_certificates=pathlib.Path("certificates/ca.crt").read_bytes(),
    )'''


if __name__ == '__main__':
    main()
