from typing import List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SequentialFeatureSelector


def seq_feat_selection(model, X_train: pd.DataFrame, y_train: pd.Series, direction: str = "backwards",
                       scoring: str = "accuracy"):
    """
    Perform sequential feature selection using SequentialFeatureSelector from sklearn.
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html#:~:text=This%20Sequential%20Feature%20Selector%20adds,validation%20score%20of%20an%20estimator.
    :param model:
    :param X_train:
    :param y_train:
    :param direction:
    :param scoring:
    :return:
    """
    # Create Sequential Feature Selector
    sfs = SequentialFeatureSelector(model,
                                    direction=direction,  # backward feature elimination
                                    scoring=scoring,
                                    cv=StratifiedKFold())

    # Fit the feature selector to training data
    sfs.fit(X_train, y_train)

    # Get selected features and feature indices
    selected_features = sfs.get_support()
    selected_feature_indices = [i for i, val in enumerate(selected_features) if val]

    print("Selected features:")
    for i, feature in enumerate(X_train.columns[selected_feature_indices]):
        print(f"Feature {i + 1}: {feature}")


def tune_knn(model, X_train: pd.DataFrame, y_train: pd.Series, X_test, y_test, param_grid) -> Tuple[
    dict, KNeighborsClassifier, float]:
    """
    Tune the hyperparameters of the K-Nearest Neighbors model using GridSearchCV.
    https://medium.com/@agrawalsam1997/hyperparameter-tuning-of-knn-classifier-a32f31af25c7
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    :param X_encoded:
    :param y_encoded:
    :param X_test:
    :param y_test:
    :return:
    """
    # Perform grid search with 5-fold cross-validation
    gscv = GridSearchCV(model, param_grid=param_grid, cv=5, verbose=1)

    gscv.fit(X_train, y_train)

    best_params = gscv.best_params_
    best_model = gscv.best_estimator_
    best_accuracy = best_model.score(X_test, y_test)

    return best_params, best_model, best_accuracy


def forward_feat_selection_hypertuning(X_encoded: pd.DataFrame, y_encoded: pd.Series) -> Tuple[List[str], dict, float]:
    """
    Forward feature selection with hyperparameter tuning for K-Nearest Neighbors.
    :param X_encoded: features dataset, with features encoded
    :param y_encoded: target dataset, with target encoded
    :return: (best subset of features, best hyperparameters, best model accuracy)
    """
    best_subset: List[str] = []
    best_params: dict = None
    best_score: float = 0.0

    remaining_features = [['age'],
                          ['education'],
                          ['workinghours'],
                          [col for col in X_encoded.columns if col.startswith('workclass')],
                          [col for col in X_encoded.columns if col.startswith('marital status')],
                          [col for col in X_encoded.columns if col.startswith('occupation')]]

    param_grid = {
        'n_neighbors': np.arange(2, 30, 1),
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }

    while remaining_features:
        subset_scores = []
        subset_params = []

        for feature_cat in remaining_features:
            # Combine the current best subset with the new feature
            current_subset = best_subset + feature_cat if best_subset else feature_cat.copy()

            X_subset = X_encoded[current_subset]
            X_train, X_val, y_train, y_val = train_test_split(X_subset, y_encoded, test_size=0.2, random_state=42)

            best_params, best_model, score = tune_knn(KNeighborsClassifier(), X_subset, y_encoded, X_val, y_val,
                                                      param_grid)

            subset_scores.append(score)
            subset_params.append(best_params)

        # Select the feature that improves performance the most
        best_index = subset_scores.index(max(subset_scores))
        best_score = subset_scores[best_index]
        best_params = subset_params[best_index]
        best_feature = remaining_features[best_index]

        # Update the best subset and remaining features
        best_subset = best_subset + best_feature if best_subset else best_feature.copy()
        del remaining_features[best_index]

        print("Best subset:", best_subset)
        print("Remaining features:", remaining_features)

    return best_subset, best_params, best_score


"""
Both functions below are heavily based on the following example from the scikit-learn documentation:
https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html#sphx-glr-auto-examples-model-selection-plot-multi-metric-evaluation-py
"""


def multi_metric_cv(model, scoring, X_train, y_train, param_grid, refit="AUC"):
    """
    Perform GridSearchCV with multiple scorers for a given model.
    :param model:
    :param scoring:
    :param X_train:
    :param y_train:
    :param param_grid:
    :param refit:
    :return:
    """
    # Initialize GridSearchCV
    gs = GridSearchCV(
        model,
        param_grid=param_grid,
        scoring=scoring,
        refit=refit,
        n_jobs=2,
        return_train_score=True,
    )

    # Fit the model
    gs.fit(X_train, y_train)
    results = gs.cv_results_

    return results


def plot_multi_score_cv_results(x_label, x_min_val, x_max_val, y_min_val, y_max_val, results, param_name, param_type,
                                scoring):
    """
    Plot the results of GridSearchCV with multiple scorers.
    :param x_label:
    :param x_min_val:
    :param x_max_val:
    :param y_min_val:
    :param y_max_val:
    :param results:
    :param param_name:
    :param param_type:
    :param scoring:
    :return:
    """
    # Plot the results
    plt.figure(figsize=(13, 13))
    plt.title("GridSearchCV evaluating using multiple scorers simultaneously", fontsize=16)

    plt.xlabel(x_label)
    plt.ylabel("Score")

    ax = plt.gca()
    ax.set_xlim(x_min_val, x_max_val - 1)
    ax.set_ylim(y_min_val, y_max_val)

    # Get the regular numpy array from the MaskedArray
    X_axis = np.array(results[param_name].data, dtype=param_type)
    colors = ["g", "b", "r", "y", "m", "k"]

    for scorer, color in zip(sorted(scoring), colors[:len(scoring)]):
        for sample, style in (("train", "--"), ("test", "-")):
            sample_score_mean = results["mean_%s_%s" % (sample, scorer)]
            sample_score_std = results["std_%s_%s" % (sample, scorer)]
            ax.fill_between(
                X_axis,
                sample_score_mean - sample_score_std,
                sample_score_mean + sample_score_std,
                alpha=0.1 if sample == "test" else 0,
                color=color,
            )
            ax.plot(
                X_axis,
                sample_score_mean,
                style,
                color=color,
                alpha=1 if sample == "test" else 0.7,
                label="%s (%s)" % (scorer, sample),
            )

        best_index = np.nonzero(results["rank_test_%s" % scorer] == 1)[0][0]
        best_score = results["mean_test_%s" % scorer][best_index]

        # Plot a dotted vertical line at the best score for that scorer marked by x
        ax.plot(
            [
                X_axis[best_index],
            ]
            * 2,
            [0, best_score],
            linestyle="-.",
            color=color,
            marker="x",
            markeredgewidth=3,
            ms=8,
        )

        # Annotate the best score for that scorer
        ax.annotate("%0.2f" % best_score, (X_axis[best_index], best_score + 0.005))

    plt.legend(loc="best")
    plt.grid(False)
    plt.show()