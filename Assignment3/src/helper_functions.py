import pandas as pd


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
