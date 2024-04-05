import pandas as pd
from joblib import dump, load


def load_dataset(file_path):
    try:
        dataset = pd.read_excel(file_path)
    except Exception as e:
        print(e)
        print("Error: Unable to load dataset")
        return None
    return dataset


def save_dataset(dataset, file_path):
    try:
        dataset.to_excel(file_path, index=False)
    except Exception as e:
        print(e)
        print("Error: Unable to save dataset")
        return
    return


def save_model(model, file_path):
    """
    Save the model to the file path. Make sure the file path has .joblib extension
    :param model: model object
    :param file_path: file path to save the model
    :return:
    """
    try:
        dump(model, file_path)
    except Exception as e:
        print(e)
        print("Error: Unable to save model")
        return
    return


def load_model(file_path):
    try:
        model = load(file_path)
    except Exception as e:
        print(e)
        print("Error: Unable to load model")
        return None
    return model
