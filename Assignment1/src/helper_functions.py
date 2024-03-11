import pandas as pd


def load_dataset(file_path):
    dataset = pd.read_excel(file_path)
    return dataset


def save_dataset(dataset, file_path):
    dataset.to_excel(file_path, index=False)
    return
