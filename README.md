# FeNOMan Client
FeNOMan is a federated learning-driven traffic flow classification framework that preserves data privacy while providing comparable performance to a hypothetical, centralized, collaborative ML-based traffic classification solution for networking scenarios. The framework consists of two parts. One is the server side, where the global models are hosted and advertised. Clients will subscribe to this given server and start training based on the pre-setted configuration. 

## Table of Contents
* [Main Features](#main-features)
* [Installation](#installation)
    * [Prerequisite](#prerequisite)
* [Usage](#usage)
* [Credits](#credits)
  * [Authors](#authors)
* [Acknowledgement](#acknowledgement)
* [License](#license)

## Main Features

The client instance is able to subscribe to a FeNOMan server where it can communicate via SSL encrypted channel with API key authentication. The client can query the models available on the server. It can download the models and configure it to perform prediction on data, or download and further teach it. It is important that if you want to do the teaching federated with the client, the server will always lead the process and the client cannot initiate the teaching.
The client is capable of monitoring network hardware and learning from it in real time. Thus, it can be used to teach global models in a continuous incremental manner using federated paradigms.

## Installation
Because of network card monitoring, the user must have super-user privileges to start the server. For this reason, it is necessary to install the required libraries by logging in as a super-user. Which can be done as follows:
```
sudo su
pip3 install -r requirements.txt
```

### Prerequisite
* Linux based operating system that supports NFStream
* Python 3.9

## Usage
After a successful installation, be sure to adjust the configuration files to ensure that the solution works for your needs.
The following configuration files modify the following parameters:
* **client_configuration**
  * *VALIDATION_SPLIT* - *Percentage ratio of teaching and test data to resolution.*
  * *EVALUATE_BATCH_SIZE* - *The batch size of the model used in the evaluation.*
  * *TRAIN_BATCH_SIZE* - *The batch size of the model used in the train.*
  * *TRAIN_EPOCHS* - *The number of epochs associated with training the model.*
* **core_configuration**
  * *SERVER_PROTOCOL* - *HTTP web schema which can be HTTP or HTTPS depending on whether encryption is used.*
  * *URI* - *The basic IP address where to listen on the web server. May be empty if the scan is properly configured for the target device in nfstream_configuration.*
  * *BASE_URI* - *For the web server, the API version number defined in the URLs.*
  * *CORE_PORT* - *Port number of the web server running in the background.*
  * *FENOMAN_CLIENT_PORT* - *Servers port where to listen to the Flower clients. The web server and flower server ports cannot be the same!*
  * *SECURE_MODE* - *This enables the secure SSL connection between client and server.*
  * *OCM_APIM_KEY* - *Key value associated with api key authentication for API endpoints.*
* **data_configuration**
  * *DATA_URI* - *In the case of an input file which may have a .csv or .pcap extension, the path.*
  * *DROP_VARIABLES* - *The field values to be discarded from the measurement data. These names are listed in this parameter.*
  * *TARGET_VARIABLE* - *Name of the target variable.*
  * *TRAIN_VALIDATION_SPLIT* - *Percentage ratio of teaching and test data to resolution.*
  * *N_FEATURES* - *The number of identifiable featurs in the data set that are used in the model training.*
  * *REDUCE_REGEX_VARIABLES* - *For fields with names specified in the list, regular expression matching is used to filter the data.*
* **nfstream_configuration**
  * *SOURCE* - *Network target device that can be monitored by the system in case there is no file-based input to the solution.*
  * *STATISTICAL_ANALYSIS* - *Statistical analysis applied to the sampled streams.*
  * *SPLT_ANALYSIS* - *For the analysis of early flow characteristics, for how many data sets should the system perform the analysis of early flow characteristics.*
  * *COLUMNS_TO_ANONYMIZE* - *The names of the columns to which anomalisation should be applied during the capture.*
  * *MAX_NFLOWS* - *The maximum number of flows that the procedure will sample from the target device using NFStream.*

After a successful configuration, we need to instantiate the Core class and use its functions in our verification as follows.
Initialization function for FeNOMan client instantiation.

In order to use the FeNOMan client we need to instantiate it with core.Core(). You can of course pass in the CSV data uri from which the solution will work, but this can also be left empty. The available models are retrieved with get_models(), and then the selected model (chosen from the list returned) can be set with set_model().
```python
core = core.Core()
_, available_models = core.get_models()
chosen_model = available_models[0]
_, _ = core.set_model(chosen_model)
```

In case you want to make predictions locally, predictions are generated by calling predict() on the pandas DataFrame after the set_model() procedure.
```python
prediction_data = pd.DataFrame()
core.predict(prediction_data)
```

If you want to do federated learning, you can do it with .train().
```python
core.train()
```

These steps are shown in the example.py file.

## Credits
### Authors
* Adrian Pekar
* Zied Aouini
* Laszlo Arpad Makara
* Gergely Biczok
* Balint Bicski
* Marcell Szoby
* Mate Simko

## Acknowledgement
Supported by the GÉANT Innovation Programme 2022.

## License
This project is licensed under the LGPLv3 license - see the [License](LICENSE) file for details.
