import pandas as pd
from joblib import dump, load

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
    features_df = dataset.drop(columns=[target_column])
    target_df = dataset[target_column]
    return features_df, target_df


def encode_nominal_features(dataset, nominal_features_lc, nominal_features_hc) -> pd.DataFrame:
    """
    Encode nominal features using multiple encoding techniques.
    :param dataset: dataset
    :param nominal_features_lc: nominal features with low cardinality
    :param nominal_features_hc: nominal features with high cardinality
    :return: encoded dataset
    """

    # Initialize encoders
    one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    label_encoder = LabelEncoder()

    X_encoded = dataset.copy()

    # Encode nominal features with high cardinality using label encoding
    nominal_encoded_hc = label_encoder.fit_transform(dataset[nominal_features_hc])
    nominal_encoded_hc_df = pd.DataFrame(nominal_encoded_hc,
                                         columns=[f'{feature}_encoded' for feature in nominal_features_hc])

    # Encode nominal features with low cardinality using one-hot encoding
    nominal_encoded_lc = one_hot_encoder.fit_transform(dataset[nominal_features_lc])
    nominal_encoded_df_lc = pd.DataFrame(nominal_encoded_lc,
                                         columns=one_hot_encoder.get_feature_names_out(nominal_features_lc))

    # Concatenate the encoded features to the resulting DataFrame and drop the original nominal features
    X_encoded = pd.concat([X_encoded, nominal_encoded_hc_df, nominal_encoded_df_lc], axis=1).drop(
        columns=nominal_features_lc + nominal_features_hc)

    return X_encoded
