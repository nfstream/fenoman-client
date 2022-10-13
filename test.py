import core


if __name__ == '__main__':
    new_data_uri = f'./data/comnet14-flows-part-1.csv'

    core = core.Core(new_data_uri)
    _, available_models = core.get_models()

    chosen_model = available_models[0]

    _, _ = core.set_model(chosen_model)

    #prediction_data = pd.DataFrame()
    #core.predict(prediction_data)

    core.train()

"""
1 - CSV nelkul is tudjon mukodni. Szoval live networkbol fel tudjon allni az egesz rendszer
2 - Uj CSV-vel meg nem megy. Az mar maskepp van preprocessalva, illetva mas is a DS jellege
3 - CORE.PY hibakat dob. Data osztalyra hivatkozva.
"""