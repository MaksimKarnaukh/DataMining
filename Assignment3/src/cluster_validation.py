"""
This module contains functions to validate clustering results.

For more information see the following resources:
https://piazza.com/class_profile/get_resource/ircffd3731a4n0/iu5o8ptzxdz5ev
https://www.cs.kent.edu/~jin/DM08/ClusterValidation.pdf
"""

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt


def cv_correlation(data, clusters, metric='euclidean', correlation_method='corrcoef') -> float:
    """
    Calculate the correlation of incidence and proximity matrices for the clustering of the data.
    :param data: numpy representation of the dataframe
    :param clusters: cluster labels
    :param metric: distance metric
    :param correlation_method: method to calculate the correlation ('spearman' or 'corrcoef')
    :return: correlation between pairwise distances and incidence of the same cluster
    """
    if hasattr(data, "toarray") and not isinstance(data, np.ndarray):
        data = data.toarray()

    # compute pairwise distances
    pairwise_distances: np.ndarray = pdist(data, metric=metric)  # One vector with all 'pairwise distances'
    distance_matrix: np.ndarray = squareform(pairwise_distances)  # Matrix with pairwise distances

    # the incidence matrix is a matrix where the entry (i, j) is 1 if the points i and j are in the same cluster
    n_samples = data.shape[0]
    incidence_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            if clusters[i] == clusters[j]:
                incidence_matrix[i, j] = 1
                incidence_matrix[j, i] = 1

    # Flatten the upper triangle of both matrices to calculate the correlation
    # Since the matrices are symmetric, only the ocrrelation between n(n-1)/2 entries needs to be calculated.
    upper_triangle_indices = np.triu_indices(n_samples, k=1)
    flat_distances = distance_matrix[upper_triangle_indices]
    flat_incidence = incidence_matrix[upper_triangle_indices]

    if correlation_method == 'spearman':
        correlation, _ = spearmanr(flat_distances, flat_incidence)
    elif correlation_method == 'corrcoef': # gives different results than spearmanr
        correlation = np.corrcoef(flat_distances, flat_incidence)[0, 1]
    else:
        raise ValueError('Invalid correlation method. Please choose "spearman" or "corrcoef".')

    return correlation


def cv_similarity_matrix(data, clusters, metric='euclidean') -> None:
    """
    Plot the similarity matrix ordered by cluster labels.
    :param data: numpy representation of the dataframe
    :param clusters: cluster labels
    :param metric: distance metric
    :return:
    """
    similarity_matrix: np.ndarray = 1 - pairwise_distances(data, metric=metric)
    sorted_indices: np.ndarray = np.argsort(clusters)  # Order the matrix by cluster labels
    sorted_similarity_matrix: np.ndarray = similarity_matrix[sorted_indices, :][:, sorted_indices]

    # plot similarity matrix
    plt.figure(figsize=(8, 8))
    im: plt.AxesImage = plt.imshow(sorted_similarity_matrix, cmap='inferno', interpolation='nearest')

    # colorbar
    cbar: plt.Colorbar = plt.colorbar(im)
    cbar.set_label('Similarity', loc='center')

    # we add lines to separate clusters
    unique_clusters, cluster_counts = np.unique(clusters, return_counts=True)
    cluster_boundaries: np.ndarray = np.cumsum(cluster_counts)
    for boundary in cluster_boundaries[:-1]:
        plt.axhline(boundary - 0.5, color='white', linestyle='--')
        plt.axvline(boundary - 0.5, color='white', linestyle='--')

    plt.title('Similarity Matrix Ordered by Cluster Labels')
    plt.xlabel('Points')
    plt.ylabel('Points')
    plt.show()


def cv_statistics(actual_model: any, actual_clusters: np.ndarray, data: pd.DataFrame, n_clusters: int, metric='euclidean',
               random_state=42, n_permutations=100, verbose=False) -> tuple[float, float, float, bool, bool, bool]:
    """
    Perform cluster validity statistics by comparing the actual clustering results to random permutations.

    Note: a score's atypicality will also be returned as False, if the score is greater than the mean while it should be less and vice versa.

    :param actual_model: the actual clustering model fitted on the regular data
    :param actual_clusters: the actual cluster labels from the actual model
    :param data: the data used for clustering
    :param n_clusters: the number of clusters
    :param metric: the distance metric used for clustering
    :param random_state: random state for reproducibility
    :param n_permutations: the number of random permutations to generate
    :return: z-scores and atypicality of SSE, silhouette score and correlation
    """
    # scores of the actual clustering results
    actual_sse: float = actual_model.inertia_
    actual_silhouette: float = silhouette_score(data, actual_clusters)
    actual_correlation: float = cv_correlation(data, actual_clusters, metric=metric)

    # Get the parameters of the actual model, so we can use them for the randomly permuted data model
    model_params = actual_model.get_params()

    # Generate random permutations and compute scores (SSE, silhouette score and correlation)
    sse_random: list[float] = []
    silhouette_random: list[float] = []
    correlation_random: list[float] = []
    for _ in range(n_permutations):
        if verbose:
            print(f'Permutation {_ + 1}/{n_permutations}')
        permuted_data: np.ndarray = np.apply_along_axis(np.random.permutation, 0, data)
        model_random = actual_model.__class__(
            **model_params)  # we call this model random because we will fit it with (randomly) permuted data
        random_clusters: np.ndarray = model_random.fit_predict(permuted_data)
        sse_random.append(model_random.inertia_)
        silhouette_random.append(silhouette_score(permuted_data, random_clusters))
        correlation_random.append(cv_correlation(permuted_data, random_clusters, metric=metric))

    sse_random: np.ndarray = np.array(sse_random)
    silhouette_random: np.ndarray = np.array(silhouette_random)
    correlation_random: np.ndarray = np.array(correlation_random)

    # Calculate mean and standard deviation of the random scores
    sse_mean: float = sse_random.mean()
    sse_std: float = sse_random.std()
    silhouette_mean: float = silhouette_random.mean()
    silhouette_std: float = silhouette_random.std()
    correlation_mean: float = correlation_random.mean()
    correlation_std: float = correlation_random.std()

    # Determine if the actual clustering results are atypical
    sse_z_score: float = (actual_sse - sse_mean) / sse_std
    silhouette_z_score: float = (actual_silhouette - silhouette_mean) / silhouette_std
    correlation_z_score: float = (actual_correlation - correlation_mean) / correlation_std

    sse_is_atypical: bool = sse_z_score < -2  # we want a negative z-score for sse, which then means it is below the mean
    silhouette_is_atypical: bool = silhouette_z_score > 2  # we want a positive z-score for silhouette score, which then means it is above the mean
    correlation_is_atypical: bool = correlation_z_score < -2  # we want a negative z-score for correlation, which then means it is below the mean

    print(f'---')
    print(f'Actual SSE: {actual_sse:.4f}')
    print(f'Random SSE: Mean: {sse_mean:.4f}, Std: {sse_std:.4f}')
    print(f'SSE Z-Score: {sse_z_score:.4f} => SSE is Atypical: {sse_is_atypical}')
    print(f'---')
    print(f'Actual Silhouette Score: {actual_silhouette:.4f}')
    print(f'Random Silhouette: Mean: {silhouette_mean:.4f}, Std: {silhouette_std:.4f}')
    print(f'Silhouette Z-Score: {silhouette_z_score:.4f} => Silhouette Score is Atypical: {silhouette_is_atypical}')
    print(f'---')
    print(f'Actual Correlation: {actual_correlation:.4f}')
    print(f'Random Correlation: Mean: {correlation_mean:.4f}, Std: {correlation_std:.4f}')
    print(f'Correlation Z-Score: {correlation_z_score:.4f} => Correlation is Atypical: {correlation_is_atypical}')
    print(f'---')

    return sse_z_score, silhouette_z_score, correlation_z_score, sse_is_atypical, silhouette_is_atypical, correlation_is_atypical