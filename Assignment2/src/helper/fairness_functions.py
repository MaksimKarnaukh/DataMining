import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.neighbors import KNeighborsClassifier
from aif360.sklearn.metrics import disparate_impact_ratio, statistical_parity_difference


def statistical_parity(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, sensitive_attr: str, use_lib_implementation: bool = False) -> Tuple[
    float, float]:
    """
    Calculate the Disparate Impact (DI) metric and Discrimination Score (DS) metric for a dataset.

    The theory behind this comes from the following paper:
    "Increasing fairness in supervised classification : a study of simple decorrelation methods applied to the logistic regression", by de Schaetzen Constantin.

    The implementation follows https://jp-tok.dataplatform.cloud.ibm.com/docs/content/wsj/model/wos-disparate-impact.html?context=cpdaas

    Interpretation for Disparate Impact (DI) metric:
    - If disparate impact = 1, outcomes are equally favorable for both groups.
    - If disparate impact > 1, one group has more favorable outcomes than the other.
    - If disparate impact < 1, the opposite is true.

    Interpretation for Discrimination Score (DS) metric:
    - A negative DS value indicates that the privileged group has more favorable outcomes compared to the unprivileged group.
    - A positive DS value indicates that the unprivileged group has more favorable outcomes compared to the privileged group.
    - A value of 0, both groups have equal benefit.
    - The magnitude of the DS indicates the extent of the difference in positive outcomes between the groups. A larger magnitude suggests a larger disparity in outcomes.

    :param X_test: test dataset
    :param sensitive_attr: the name of the single sensitive attribute in the dataset
    :param use_lib_implementation: whether to use the aif360 library implementation for the disparate impact ratio and statistical parity difference calculation
    :return:
    """

    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)

    y_pred = knn_model.predict(X_test)

    if use_lib_implementation:
        disparate_impact = disparate_impact_ratio(y_true=y_test, y_pred=y_pred, prot_attr=X_test[sensitive_attr])
        stat_par_diff = statistical_parity_difference(y_true=y_test, y_pred=y_pred, prot_attr=X_test[sensitive_attr])
        return disparate_impact, stat_par_diff

    # calculate Disparate Impact (DI) metric:
    # DI = (num_positives(priviliged=False) / num_instances(priviliged=False)) / (num_positives(priviliged=True) / num_instances(priviliged=True))
    num_instances_priv_false = X_test[X_test[sensitive_attr] == 0].shape[0]
    num_instances_priv_true = X_test[X_test[sensitive_attr] == 1].shape[0]

    num_positives_priv_false = np.sum((y_pred == 1) & (X_test[sensitive_attr] == 0))
    num_positives_priv_true = np.sum((y_pred == 1) & (X_test[sensitive_attr] == 1))

    DI = (num_positives_priv_false / num_instances_priv_false) / (num_positives_priv_true / num_instances_priv_true)

    # calculate Discrimination Score (DS) metric:
    # DS = num_positives(priviliged=False) / num_instances(priviliged=False) - num_positives(priviliged=True) / num_instances(priviliged=True)
    DS = num_positives_priv_false / num_instances_priv_false - num_positives_priv_true / num_instances_priv_true

    return DI, DS
