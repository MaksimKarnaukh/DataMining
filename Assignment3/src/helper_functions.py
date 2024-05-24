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


# save cluster labels to file
def save_cluster_labels(data, cluster_labels, filename):
    """
    Save the cluster labels to a file.
    :param data: original data dataframe
    :param cluster_labels: cluster labels
    :param filename: output filename
    :return:
    """
    data['cluster'] = cluster_labels
    data.to_excel(filename, index=False)
