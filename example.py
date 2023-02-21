import json
import core
import pandas as pd


if __name__ == '__main__':
    core = core.Core()
    _, available_models = core.get_models()

    print(json.dumps(available_models, indent=4))

    chosen_model = available_models[0]

    _, _ = core.set_model(chosen_model)

    prediction_data = r'.\data\comnet14-flows-part-1.csv'
    predictions = core.predict(prediction_data)

    decoded_predictions = core.get_decode_classes(predictions)

    df = pd.read_csv(prediction_data, low_memory=False)
    counter = 0
    for index, row in df.iterrows():
        if counter == 15:
            break

        print(f"Actual application: {row['application_name']} predicted: {decoded_predictions[index]}")
        counter += 1

    _, _ = core.train()
