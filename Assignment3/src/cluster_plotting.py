"""
This module contains functions to plot cluster results.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.neighbors import NearestNeighbors


def elbow_method_plot(model, tfidf_features, max_k, use_own_function=False, distance_metric='euclidean') -> None:
    """
    Plots the Elbow Method graph for optimal k in KMeans clustering.
    :param model: model to use for clustering
    :param tfidf_features: TF-IDF features
    :param max_k: maximum number of clusters to consider
    :param use_own_function: whether to use the custom implementation or the Yellowbrick implementation
    :param distance_metric: distance metric to use
    :return:
    """
    if use_own_function:
        sse = []  # List to store SSE values for different k

        for k in range(1, max_k + 1):
            m = model.__class__(n_clusters=k, random_state=42)
            m.fit(tfidf_features)
            sse.append(m.inertia_)

        # Plot SSE vs. number of clusters (k)
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, max_k + 1), sse, marker='o')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Sum of Squared Errors (SSE)')
        plt.title('Elbow Method for Optimal k')
        plt.xticks(range(1, max_k + 1))
        plt.grid(True)
        plt.show()
    else:
        model = model.__class__(random_state=42)
        visualizer = KElbowVisualizer(model, k=(1, max_k + 1), distance_metric=distance_metric)
        visualizer.fit(tfidf_features)
        visualizer.show()
        plt.show()


def plot_clusters_2d_all(text_data, clusters, n_components=2, random_state=42) -> None:
    """
    Plot the clusters in 2D using PCA, SVD, and t-SNE.
    :param text_data: vectorized text data
    :param clusters: cluster labels
    :param n_components: number of components for PCA, SVD, and t-SNE
    :param random_state: random state for PCA, SVD, and t-SNE
    :return:
    """

    n_clusters = len(set(clusters))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'tab:orange', 'tab:olive', 'tab:pink', 'tab:brown'] # [cm.tab10(i/float(n_clusters)) for i in range(n_clusters)]

    # apply PCA to reduce the dimensionality to 2d
    pca = PCA(n_components=n_components, random_state=random_state)
    if isinstance(text_data, np.ndarray):
        reduced_features_pca = pca.fit_transform(text_data)
    else:
        reduced_features_pca = pca.fit_transform(text_data.toarray())

    # apply TruncatedSVD to reduce the dimensionality to 2d
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    reduced_features_svd = svd.fit_transform(text_data)

    # apply t-SNE to reduce the dimensionality to 2d
    tsne = TSNE(n_components=n_components, init="random", random_state=random_state)
    reduced_features_tsne = tsne.fit_transform(text_data)

    f, ax = plt.subplots(1, 3, figsize=(21, 6))

    # plot PCA results
    for i in range(n_clusters):
        ax[0].scatter(reduced_features_pca[clusters == i, 0], reduced_features_pca[clusters == i, 1],
                      color=colors[i], label=f'Cluster {i}')
    ax[0].set_title('PCA Cluster Plot')
    ax[0].set_xlabel('Principal Component 1')
    ax[0].set_ylabel('Principal Component 2')

    # plot SVD results
    for i in range(n_clusters):
        ax[1].scatter(reduced_features_svd[clusters == i, 0], reduced_features_svd[clusters == i, 1],
                      color=colors[i], label=f'Cluster {i}')
    ax[1].set_title('SVD Cluster Plot')
    ax[1].set_xlabel('Component 1')
    ax[1].set_ylabel('Component 2')

    # Plot t-SNE results
    for i in range(n_clusters):
        ax[2].scatter(reduced_features_tsne[clusters == i, 0], reduced_features_tsne[clusters == i, 1],
                      color=colors[i], label=f'Cluster {i}')
    ax[2].set_title('t-SNE Cluster Plot')
    ax[2].set_xlabel('Component 1')
    ax[2].set_ylabel('Component 2')

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()

    plt.show()


def plot_clusters_3d_all(text_data, clusters, n_components=3, random_state=42) -> None:
    """
    Plot the clusters in 3D using PCA, SVD, and t-SNE.
    :param text_data: vectorized text data
    :param clusters: cluster labels
    :param n_components: number of components for PCA, SVD, and t-SNE
    :param random_state: random state for PCA, SVD, and t-SNE
    :return:
    """

    n_clusters = len(set(clusters))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'tab:orange', 'tab:olive', 'tab:pink', 'tab:brown'] # [cm.tab10(i/float(n_clusters)) for i in range(n_clusters)]

    # apply PCA to reduce the dimensionality to 3d
    pca = PCA(n_components=n_components, random_state=random_state)
    if isinstance(text_data, np.ndarray):
        reduced_features_pca = pca.fit_transform(text_data)
    else:
        reduced_features_pca = pca.fit_transform(text_data.toarray())

    # apply TruncatedSVD to reduce the dimensionality to 3d
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    reduced_features_svd = svd.fit_transform(text_data)

    # apply t-SNE to reduce the dimensionality to 3d
    tsne = TSNE(n_components=n_components, init="random", random_state=random_state)
    reduced_features_tsne = tsne.fit_transform(text_data)

    fig = plt.figure(figsize=(18, 6))

    # plot PCA results in 3D
    ax1 = fig.add_subplot(131, projection='3d')
    for i in range(n_clusters):
        ax1.scatter(reduced_features_pca[clusters == i, 0], reduced_features_pca[clusters == i, 1],
                    reduced_features_pca[clusters == i, 2], color=colors[i], label=f'Cluster {i}')
    ax1.set_title('PCA Cluster Plot')
    ax1.set_xlabel('Principal Component 1')
    ax1.set_ylabel('Principal Component 2')
    ax1.set_zlabel('Principal Component 3')

    # plot SVD results in 3D
    ax2 = fig.add_subplot(132, projection='3d')
    for i in range(n_clusters):
        ax2.scatter(reduced_features_svd[clusters == i, 0], reduced_features_svd[clusters == i, 1],
                    reduced_features_svd[clusters == i, 2], color=colors[i], label=f'Cluster {i}')
    ax2.set_title('SVD Cluster Plot')
    ax2.set_xlabel('Component 1')
    ax2.set_ylabel('Component 2')
    ax2.set_zlabel('Component 3')

    # plot t-SNE results in 3D
    ax3 = fig.add_subplot(133, projection='3d')
    for i in range(n_clusters):
        ax3.scatter(reduced_features_tsne[clusters == i, 0], reduced_features_tsne[clusters == i, 1],
                    reduced_features_tsne[clusters == i, 2], color=colors[i], label=f'Cluster {i}')
    ax3.set_title('t-SNE Cluster Plot')
    ax3.set_xlabel('Component 1')
    ax3.set_ylabel('Component 2')
    ax3.set_zlabel('Component 3')

    ax1.legend()
    ax2.legend()
    ax3.legend()
    plt.show()


def plot_k_distance_graph(text_data: np.ndarray, k: int = 4) -> None:
    """
    Plot the k-distance graph for DBSCAN to help determine the optimal epsilon value (roughly look for the elbow on the right side of the graph).

    See also https://digitaldwellings.tech/blog/article/33/

    :param text_data: The input data in the form of a (TF-IDF) matrix or array.
    :param k: The number of neighbors to consider (default is 4).
    """
    if not isinstance(text_data, np.ndarray):
        text_data = text_data.toarray()

    nbrs = NearestNeighbors(n_neighbors=k).fit(text_data)
    distances, indices = nbrs.kneighbors(text_data)

    # sort the distances to the k-th nearest neighbor in ascending order
    k_distances = distances[:, k - 1]
    k_distances.sort()

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(k_distances) + 1), k_distances)
    plt.xlabel('Data Points sorted by distance to {}th nearest neighbor'.format(k))
    plt.ylabel('{}th Nearest Neighbor Distance'.format(k))
    plt.title('{}-Distance Graph (for DBSCAN)'.format(k))
    plt.grid(True)
    plt.show()
