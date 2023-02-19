import json
import core


if __name__ == '__main__':
    core = core.Core()
    _, available_models = core.get_models()

    print(json.dumps(available_models, indent=4))

    chosen_model = available_models[0]

    _, _ = core.set_model(chosen_model)

    #prediction_data = pd.DataFrame()
    #core.predict(prediction_data)

    _, _ = core.train()
