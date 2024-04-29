from typing import Tuple, Any, List

import pandas as pd
from joblib import dump, load
from pandas import DataFrame

from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def load_dataset(file_path: str) -> pd.DataFrame | None:
    try:
        dataset = pd.read_excel(file_path)
    except Exception as e:
        print(e)
        print("Error: Unable to load dataset")
        return None
    return dataset


def save_dataset(dataset: pd.DataFrame, file_path: str):
    try:
        dataset.to_excel(file_path, index=False)
    except Exception as e:
        print(e)
        print("Error: Unable to save dataset")
        return
    return


def save_model(model: any, file_path: str):
    """
    Save the model to the file path. Make sure the file path has .joblib extension.
    https://scikit-learn.org/stable/model_persistence.html
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


def load_model(file_path: str):
    try:
        model = load(file_path)
    except Exception as e:
        print(e)
        print("Error: Unable to load model")
        return None
    return model


def get_features_and_target(dataset: pd.DataFrame, target_column: str) -> tuple[
    pd.DataFrame, pd.Series | None | pd.DataFrame]:
    """
    Get features and target from the dataset.
    :param dataset: dataset
    :param target_column: target column name
    :return: (features_df, target_df)
    """
    features_df: pd.DataFrame = dataset.drop(columns=[target_column])
    target_df: pd.Series | pd.DataFrame = dataset[target_column]
    return features_df, target_df


def encode_nominal_features(dataset: pd.DataFrame, nominal_features_lc: List[str],
                            nominal_features_hc: List[str]) -> pd.DataFrame:
    """
    Encode nominal features using multiple encoding techniques.
    :param dataset: dataset
    :param nominal_features_lc: nominal features with low cardinality
    :param nominal_features_hc: nominal features with high cardinality
    :return: encoded dataset
    """

    label_encoder: LabelEncoder = LabelEncoder()

    X_encoded: pd.DataFrame = dataset.copy()

    # Encode nominal features with high cardinality using label encoding
    if nominal_features_hc:
        X_encoded[nominal_features_hc] = X_encoded[nominal_features_hc].apply(label_encoder.fit_transform)

    # Encode nominal features with low cardinality using one-hot encoding
    if nominal_features_lc:
        X_encoded = pd.get_dummies(X_encoded, columns=nominal_features_lc, dtype=int)

    return X_encoded


def encode_all_features(X: pd.DataFrame, y: pd.Series | pd.DataFrame, columns_to_exclude: list[str]) -> tuple[
    DataFrame, Any]:
    """
    Encode all features in the dataset.
    Note: nominal_features_hc has a set that is empty because the 'occupation' feature after consideration
    is not a high cardinality feature, but should you want to include it, you can add it to the set.
    :param X: features dataset
    :param y: target dataset
    :param columns_to_exclude: features to exclude from encoding
    :return: encoded X and y datasets
    """
    X_encoded: pd.DataFrame = X.copy()

    # List of nominal features
    nominal_features_lc: List[str] = list(
        {'workclass', 'marital status', 'gave birth this year', 'sex', 'occupation'} - set(
            columns_to_exclude))  # low cardinality features
    nominal_features_hc: List[str] = list(set() - set(columns_to_exclude))  # high cardinality features

    # Encoded X and y datasets
    X_encoded = encode_nominal_features(X_encoded, nominal_features_lc, nominal_features_hc)
    y_encoded: pd.Series | pd.DataFrame = y.map({'low': 0, 'high': 1})

    return X_encoded, y_encoded
