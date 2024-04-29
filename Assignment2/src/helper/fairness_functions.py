import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from typing import Tuple, Any

from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from aif360.sklearn.metrics import disparate_impact_ratio, statistical_parity_difference, equal_opportunity_difference, \
    average_odds_difference


def statistical_measures(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series,
                       sensitive_attr: str, use_lib_implementation: bool = False) -> Tuple[Any, Any, Any, Any, Any]:
    """
    Calculate the following metrics:
    -Disparate Impact (DI) metric => DI = P(Yˆ = +|Z = 0) / P(Yˆ = +|Z = 1)
    -Discrimination Score (DS) metric => DS = P(Yˆ = +|Z = 0) − P(Yˆ = +|Z = 1)
    -Equal Opportunity (EO) metric => DEO = P(Yˆ = +|Z = 1, Y = +) − P(Yˆ = +|Z = 0, Y = +)
    -Equalized Odds (EOdds) metric => EOdds = P(Yˆ = +|Z = 1, Y = −) − P(Yˆ = +|Z = 0, Y = −)

    The theory behind this comes from the following paper:
    "Increasing fairness in supervised classification : a study of simple decorrelation methods applied to the logistic regression", by de Schaetzen Constantin.

    The implementation for DI and DS follows https://jp-tok.dataplatform.cloud.ibm.com/docs/content/wsj/model/wos-disparate-impact.html?context=cpdaas

    Interpretation for Disparate Impact (DI) metric:
    - If disparate impact = 1, outcomes are equally favorable for both groups.
    - If disparate impact > 1, one group has more favorable outcomes than the other.
    - If disparate impact < 1, the opposite is true.

    Interpretation for Discrimination Score (DS) metric:
    - A negative value indicates that the privileged group has more favorable outcomes compared to the unprivileged group.
    - A positive value indicates that the unprivileged group has more favorable outcomes compared to the privileged group.
    - A value of 0, both groups have equal benefit.

    Interpretation for Equal Opportunity (EO) metric and Equalized Odds (EOdds) metric:
    - A positive value indicates that the privileged group has more favorable outcomes compared to the unprivileged group.
    - A negative value indicates that the unprivileged group has more favorable outcomes compared to the privileged group.
    - A value of 0, both groups have equal benefit.

    :param X_train: training dataset
    :param y_train: training labels
    :param X_test: test dataset
    :param y_test: test labels
    :param sensitive_attr: the name of the single sensitive attribute in the dataset
    :param use_lib_implementation: whether to use the aif360 library implementation for the disparate impact ratio and statistical parity difference calculation.
    If True, the last element in the returning tuple will be the Average Odds Difference (AOD) instead of Equalized odds.
    As far as I have checked, corresponding to the theory, the aif360 library implementation of Equal Opportunity
    is incorrect in the sense that it switches the privileged and unprivileged groups in the difference formula.
    :return: disparate impact, discrimination score, equal opportunity, equalized odds
    """

    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_test)

    conf_matrix = confusion_matrix(y_test, y_pred)

    if use_lib_implementation:
        disparate_impact = disparate_impact_ratio(y_true=y_test, y_pred=y_pred, prot_attr=X_test[sensitive_attr])
        stat_par_diff = statistical_parity_difference(y_true=y_test, y_pred=y_pred, prot_attr=X_test[sensitive_attr])
        EO = equal_opportunity_difference(y_true=y_test, y_pred=y_pred, prot_attr=X_test[sensitive_attr])
        AOD = average_odds_difference(y_true=y_test, y_pred=y_pred, prot_attr=X_test[sensitive_attr])
        return disparate_impact, stat_par_diff, EO, AOD, conf_matrix

    # calculate Disparate Impact (DI) metric and Discrimination Score (DS) metric
    num_instances_priv_false = X_test[X_test[sensitive_attr] == 0].shape[0]
    num_instances_priv_true = X_test[X_test[sensitive_attr] == 1].shape[0]
    num_positives_priv_false = np.sum((y_pred == 1) & (X_test[sensitive_attr] == 0))
    num_positives_priv_true = np.sum((y_pred == 1) & (X_test[sensitive_attr] == 1))

    DI = (num_positives_priv_false / num_instances_priv_false) / (num_positives_priv_true / num_instances_priv_true)
    DS = num_positives_priv_false / num_instances_priv_false - num_positives_priv_true / num_instances_priv_true

    # calculate Equal Opportunity (EO) metric and Equalized Odds (EOdds) metric
    pos_idx = y_test == 1
    neg_idx = y_test == 0

    tp_privileged_pos = np.sum((y_test[pos_idx] == 1) & (y_pred[pos_idx] == 1) & (X_test[sensitive_attr][pos_idx] == 1))
    tp_unprivileged_pos = np.sum((y_test[pos_idx] == 1) & (y_pred[pos_idx] == 1) & (X_test[sensitive_attr][pos_idx] == 0))
    fp_privileged_neg = np.sum((y_test[neg_idx] == 0) & (y_pred[neg_idx] == 1) & (X_test[sensitive_attr][neg_idx] == 1))
    fp_unprivileged_neg = np.sum(
        (y_test[neg_idx] == 0) & (y_pred[neg_idx] == 1) & (X_test[sensitive_attr][neg_idx] == 0))

    EO = (tp_privileged_pos / np.sum((X_test[sensitive_attr] == 1) & (y_test[pos_idx] == 1)) -
          tp_unprivileged_pos / np.sum((X_test[sensitive_attr] == 0) & (y_test[pos_idx] == 1)))
    EOdds = (fp_privileged_neg / np.sum((X_test[sensitive_attr] == 1) & (y_test[neg_idx] == 0)) -
             fp_unprivileged_neg / np.sum((X_test[sensitive_attr] == 0) & (y_test[neg_idx] == 0)))

    return DI, DS, EO, EOdds, conf_matrix


def print_conf_matrix(conf_matrix):
    total = conf_matrix.sum()
    tp = conf_matrix[1, 1]
    tn = conf_matrix[0, 0]
    fp = conf_matrix[0, 1]
    fn = conf_matrix[1, 0]

    # # Print percentages with labels
    print("\nConfusion Matrix: ")
    print("Percentage of True positives =", (tp / total))
    print("Percentage of True negatives =", (tn / total))
    print("Percentage of False positives =", (fp / total))
    print("Percentage of False negatives =", (fn / total))

    print("FPR: ", fp / (fp + tn))
    print("TPR: ", tp / (tp + fn))
    print("PPP: ", tp / (tp + fp))
