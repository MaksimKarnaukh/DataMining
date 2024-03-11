import pandas as pd


def load_dataset(file_path):
    dataset = pd.read_excel(file_path)
    return dataset
