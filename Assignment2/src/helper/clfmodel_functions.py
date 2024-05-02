from typing import List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import make_scorer, accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.feature_selection import SequentialFeatureSelector

from .fairness_functions import split_male_female_metrics, calculate_composite_metric


def seq_feat_selection(model: any, X_train: pd.DataFrame, y_train: pd.Series, direction: str = "backward",
                       scoring: str = "accuracy", sfs_tol: float = 0.0001) -> None:
    """
    Perform sequential feature selection using SequentialFeatureSelector from sklearn.
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html#:~:text=This%20Sequential%20Feature%20Selector%20adds,validation%20score%20of%20an%20estimator.
    :param model: Classifier model
    :param X_train: features dataset, with features encoded
    :param y_train: target dataset, with target encoded
    :param direction: Direction of feature selection. Can be "forward" or "backward".
    :param scoring: Scoring metric to use for feature selection.
    :param sfs_tol: Tolerance for the stopping criterion.
    :return:
    """
    sfs: SequentialFeatureSelector = SequentialFeatureSelector(model,
                                                               direction=direction,
                                                               scoring=scoring,
                                                               cv=StratifiedKFold(),
                                                               tol=sfs_tol)

    sfs.fit(X_train, y_train)

    selected_features = sfs.get_support()
    selected_feature_indices = [i for i, val in enumerate(selected_features) if val]

    print("Selected features:")
    for i, feature in enumerate(X_train.columns[selected_feature_indices]):
        print(f"Feature {i + 1}: {feature}")


def tune_model(model: any, X_encoded: pd.DataFrame, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series,
               param_grid: dict, top_n: int = 10, cv: int = 5, ensure_fairness: bool = False) -> Tuple[
    dict, any, float]:
    """
    Tune the hyperparameters of a model using GridSearchCV.
    https://medium.com/@agrawalsam1997/hyperparameter-tuning-of-knn-classifier-a32f31af25c7
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
    :param model: Classifier model
    :param X_encoded: features dataset, with features encoded
    :param X_train: features dataset, with features encoded
    :param y_train: target dataset, with target encoded
    :param X_test: test features dataset, with features encoded
    :param y_test: test target dataset, with target encoded
    :param param_grid: Dictionary of hyperparameters to tune
    :param top_n: Number of top-performing models from which to select the best model based on test set accuracy (or composite metric). Default is 10.
    :param cv: Number of folds for cross-validation. Default is 5.
    :param ensure_fairness: Whether to ensure fairness in the model. Default is False.
    :return:
    """
    # Perform grid search with 5-fold cross-validation
    gscv: GridSearchCV = GridSearchCV(model, param_grid=param_grid, cv=cv, verbose=1)
    gscv.fit(X_train, y_train)

    top_n_results_idx = np.argsort(gscv.cv_results_['mean_test_score'])[-top_n:]

    best_params: dict = dict()
    best_model: any = None
    best_accuracy: float = 0.0
    best_composite_metric = 0.0  # best composite metric for balance between fairness and accuracy

    if not ensure_fairness:  # the standard case, this was the original function
        # Evaluate the top N models on the test set
        for i in top_n_results_idx:
            model = model.__class__()
            model.set_params(**gscv.cv_results_['params'][i])
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            if accuracy > best_accuracy:
                best_params = gscv.cv_results_['params'][i]
                best_model = model
                best_accuracy = accuracy

        return best_params, best_model, best_accuracy

    else:  # the new case, where we try to ensure some fairness
        # Evaluate the top N models on the test set
        for i in np.argsort(gscv.cv_results_['mean_test_score'])[-30:]:
            model = model.__class__()
            model.set_params(**gscv.cv_results_['params'][i])
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            fpr_male, fpr_female, tpr_male, tpr_female = None, None, None, None
            if ensure_fairness:
                X_male_test = X_test[X_encoded['sex_Male'] == 1]
                y_male_test = y_test[X_encoded['sex_Male'] == 1]
                X_female_test = X_test[X_encoded['sex_Male'] == 0]
                y_female_test = y_test[X_encoded['sex_Male'] == 0]
                split_testsets = [X_male_test, y_male_test, X_female_test, y_female_test]
                fpr_male, fpr_female, tpr_male, tpr_female = split_male_female_metrics(model, X_test, y_test, split_testsets=split_testsets)
            else:
                fpr_male, fpr_female, tpr_male, tpr_female = split_male_female_metrics(model, X_test, y_test)

            # Calculate composite metric
            composite_metric = calculate_composite_metric(accuracy, fpr_male, fpr_female, tpr_male, tpr_female)

            # Update best_params, best_model, and best_accuracy if composite_metric is better
            if composite_metric > best_composite_metric:
                best_params = gscv.cv_results_['params'][i]
                best_model = model
                best_accuracy = accuracy
                best_composite_metric = composite_metric

        return best_params, best_model, best_accuracy


def forward_feat_selection_hypertuning(model: any, param_grid: dict, X_train: pd.DataFrame, y_train: pd.Series,
                                       X_val: pd.DataFrame, y_val: pd.Series, epsilon: float = 1e-3,
                                       ensure_fairness: bool = False, columns_to_exclude: list = None) -> Tuple[List[str], dict, float]:
    """
    Forward feature selection with hyperparameter tuning for model.
    :param model: Classifier model
    :param param_grid: Dictionary of hyperparameters to tune
    :param X_train: features dataset, with features encoded
    :param y_train: target dataset, with target encoded
    :param X_val: validation features dataset, with features encoded
    :param y_val: validation target dataset, with target encoded
    :param epsilon: threshold for improvement in accuracy
    :param ensure_fairness: Whether to ensure fairness in the model. Default is False.
    :param columns_to_exclude: columns to exclude from feature selection
    :return: (best subset of features, best hyperparameters, best model accuracy)
    """
    best_subset: List[str] = []
    best_params: dict = None
    best_score: float = 0.0

    if columns_to_exclude is None:
        columns_to_exclude = []

    remaining_features = [list({'age'} - set(columns_to_exclude)),
                          list({'education'} - set(columns_to_exclude)),
                          list({'workinghours'} - set(columns_to_exclude)),
                          list({'ability to speak english'} - set(columns_to_exclude)),
                          [col for col in X_train.columns if col.startswith('workclass')],
                          [col for col in X_train.columns if col.startswith('marital status')],
                          [col for col in X_train.columns if col.startswith('occupation')],
                          [col for col in X_train.columns if col.startswith('sex')],
                          [col for col in X_train.columns if col.startswith('gave birth this year')]]

    if ensure_fairness:
        if 'sex' in columns_to_exclude:
            print("Cannot ensure fairness (use composite metric) without the 'sex' feature. Exiting.")
            return best_subset, best_params, best_score
        best_subset = [col for col in X_train.columns if col.startswith('sex')]
        # remove the sex feature(s) from remaining_features
        remaining_features.pop(7)

    # remove all empty lists from remaining_features
    remaining_features = [feature_cat for feature_cat in remaining_features if len(feature_cat) > 0]

    while remaining_features:
        subset_scores = []
        subset_params = []

        for feature_cat in remaining_features:
            if len(feature_cat) == 0:
                # normally, we should never enter this block
                print("Empty feature category")
                continue

            # Combine the current best subset with the new feature
            current_subset: List[str] = best_subset + feature_cat if best_subset else feature_cat.copy()

            X_subset: pd.DataFrame = X_train[current_subset]
            X_val_subset: pd.DataFrame = X_val[current_subset]

            best_params, best_model, score = tune_model(model, X_subset, y_train, X_val_subset, y_val, param_grid,
                                                        ensure_fairness=ensure_fairness)

            subset_scores.append(score)
            subset_params.append(best_params)

        # We select the feature that improves performance the most
        best_index = subset_scores.index(max(subset_scores))
        best_score_curr = subset_scores[best_index]

        if best_score_curr > best_score + epsilon:
            best_score = best_score_curr
            best_params = subset_params[best_index]
            best_feature = remaining_features[best_index]

            # We update the best subset and remaining features
            best_subset = best_subset + best_feature if best_subset else best_feature.copy()
            del remaining_features[best_index]

            print("Best subset:", best_subset)
            print("Remaining features:", remaining_features)
        else:  # If the current best feature does not improve performance, no feature after it will, so we break
            print("Best subset:", best_subset)
            print("Remaining features:", remaining_features)
            break

    return best_subset, best_params, best_score


"""
Both functions below are heavily based on the following example from the scikit-learn documentation:
https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html#sphx-glr-auto-examples-model-selection-plot-multi-metric-evaluation-py
"""


def multi_metric_cv(model: any, scoring: dict, X_train: pd.DataFrame, y_train: pd.Series, param_grid: dict,
                    refit: str = "Accuracy") -> dict:
    """
    Perform GridSearchCV with multiple scorers for a given model.
    :param model: Classifier model
    :param scoring: Dictionary of scoring metrics
    :param X_train: features dataset, with features encoded
    :param y_train: target dataset, with target encoded
    :param param_grid: Dictionary of hyperparameters to tune
    :param refit: Scoring metric to use for refitting the model. Default is "Accuracy".
    :return: results (dict)
    """
    gs: GridSearchCV = GridSearchCV(
        model,
        param_grid=param_grid,
        scoring=scoring,
        refit=refit,
        n_jobs=2,
        return_train_score=True,
    )

    gs.fit(X_train, y_train)
    return gs.cv_results_


def plot_multi_score_cv_results(x_label: str, x_min_val: float, x_max_val: float, y_min_val: float, y_max_val: float,
                                results: dict, param_name: str, param_type: any,
                                scoring: dict) -> None:
    """
    Plot the results of GridSearchCV with multiple scorers.
    :param x_label: Label for the x-axis
    :param x_min_val: Minimum value for the x-axis
    :param x_max_val: Maximum value for the x-axis
    :param y_min_val: Minimum value for the y-axis
    :param y_max_val: Maximum value for the y-axis
    :param results: Dictionary of results from GridSearchCV
    :param param_name: Name of the parameter to plot
    :param param_type: Type of the parameter to plot
    :param scoring: Dictionary of scoring metrics
    :return:
    """
    plt.figure(figsize=(10, 10))
    plt.title("GridSearchCV evaluating using multiple scorers simultaneously", fontsize=16)

    plt.xlabel(x_label)
    plt.ylabel("Score")

    ax = plt.gca()
    ax.set_xlim(x_min_val, x_max_val - 1)
    ax.set_ylim(y_min_val, y_max_val)

    X_axis = np.array(results[param_name].data, dtype=param_type)
    colors = ["g", "b", "r", "y", "m", "k"]  # don't pass more than 6 scorers

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

        ax.annotate("%0.2f" % best_score, (X_axis[best_index], best_score + 0.005))

    plt.legend(loc="best")
    plt.grid(False)
    plt.show()
