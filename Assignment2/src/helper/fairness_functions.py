import warnings
import numpy as np
import pandas as pd
from typing import Tuple, Any

from aif360.sklearn.metrics import disparate_impact_ratio, statistical_parity_difference, equal_opportunity_difference, \
    average_odds_difference
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from IPython.display import display

warnings.filterwarnings("ignore")


def statistical_measures(X_test: pd.DataFrame, y_test: pd.Series, y_pred: np.ndarray,
                         sensitive_attr: str, use_lib_implementation: bool = False) -> Tuple[float, float, float, float, Any]:
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

    :param X_test: the test set features
    :param y_test: the test set target
    :param y_pred: the predicted target
    :param sensitive_attr: the name of the single sensitive attribute in the dataset
    :param use_lib_implementation: whether to use the aif360 library implementation for the disparate impact ratio and statistical parity difference calculation.
    If True, the last element in the returning tuple will be the Average Odds Difference (AOD) instead of Equalized odds.
    As far as I have checked, corresponding to the theory, the aif360 library implementation of Equal Opportunity
    is incorrect in the sense that it switches the privileged and unprivileged groups in the difference formula.
    :return: disparate impact, discrimination score, equal opportunity, equalized odds, conf_matrix
    """

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
    tp_unprivileged_pos = np.sum(
        (y_test[pos_idx] == 1) & (y_pred[pos_idx] == 1) & (X_test[sensitive_attr][pos_idx] == 0))
    fp_privileged_neg = np.sum((y_test[neg_idx] == 0) & (y_pred[neg_idx] == 1) & (X_test[sensitive_attr][neg_idx] == 1))
    fp_unprivileged_neg = np.sum(
        (y_test[neg_idx] == 0) & (y_pred[neg_idx] == 1) & (X_test[sensitive_attr][neg_idx] == 0))

    EO = (tp_privileged_pos / np.sum((X_test[sensitive_attr] == 1) & (y_test[pos_idx] == 1)) -
          tp_unprivileged_pos / np.sum((X_test[sensitive_attr] == 0) & (y_test[pos_idx] == 1)))
    EOdds = (fp_privileged_neg / np.sum((X_test[sensitive_attr] == 1) & (y_test[neg_idx] == 0)) -
             fp_unprivileged_neg / np.sum((X_test[sensitive_attr] == 0) & (y_test[neg_idx] == 0)))

    return DI, DS, EO, EOdds, conf_matrix


def calc_fairness_metrics(conf_matrix: any) -> Tuple[float, float, float]:
    """
    Calculate the fairness metrics False Positive Rate (FPR), True Positive Rate (TPR), and Positive Predictive Parity (PPP).
    :param conf_matrix: the confusion matrix, in our case, it is (needs to be) a 2x2 matrix
    :return: FPR, TPR, PPP
    """
    tp = conf_matrix[1, 1]
    tn = conf_matrix[0, 0]
    fp = conf_matrix[0, 1]
    fn = conf_matrix[1, 0]

    fpr = fp / (fp + tn)
    tpr = tp / (tp + fn)
    ppp = tp / (tp + fp)

    return fpr, tpr, ppp


def get_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Calculate the accuracy, precision, recall, and F1-score metrics.
    :param y_true: ground truth target values
    :param y_pred: predicted target values
    :return: accuracy, precision, recall, F1-score
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return accuracy, precision, recall, f1


def split_male_female_metrics(model: any, X_test: pd.DataFrame, y_test: pd.Series, print_metrics: bool = False, split_testsets: list = None) -> Tuple[float, float, float, float]:
    """
    Get the fairness metrics FPR and TPR for males and females.
    :param model: the trained model
    :param X_test: the test set features
    :param y_test: the test set target
    :param print_metrics: set to True to print the confusion matrices and the metrics table for both sexes
    :param split_testsets: if not None, it should be a list containing the male/female split test sets.
    The list should be in the following form: [X_male_test, y_male_test, X_female_test, y_female_test].
    :return: fpr_male, fpr_female, tpr_male, tpr_female
    """

    if split_testsets is None:
        X_male_test = X_test[X_test['sex_Male'] == 1]
        y_male_test = y_test[X_test['sex_Male'] == 1]
        X_female_test = X_test[X_test['sex_Male'] == 0]
        y_female_test = y_test[X_test['sex_Male'] == 0]
    else:
        X_male_test = split_testsets[0]
        y_male_test = split_testsets[1]
        X_female_test = split_testsets[2]
        y_female_test = split_testsets[3]

    y_male_pred = model.predict(X_male_test)
    y_female_pred = model.predict(X_female_test)

    male_accuracy, male_precision, male_recall, male_f1 = get_metrics(y_male_test, y_male_pred)
    female_accuracy, female_precision, female_recall, female_f1 = get_metrics(y_female_test, y_female_pred)

    conf_matrix_male = confusion_matrix(y_male_test, y_male_pred)
    conf_matrix_female = confusion_matrix(y_female_test, y_female_pred)

    fpr_male, tpr_male, ppp_male = calc_fairness_metrics(conf_matrix_male)
    fpr_female, tpr_female, ppp_female = calc_fairness_metrics(conf_matrix_female)

    if print_metrics:
        print("\nConfusion Matrix for Male group:")
        print_conf_matrix(conf_matrix_male)
        print("\nConfusion Matrix for Female group:")
        print_conf_matrix(conf_matrix_female)

        # Compare the model's accuracy for male and female groups
        metrics_table = pd.DataFrame({
            'Gender': ['Male', 'Female'],
            'Accuracy': [male_accuracy, female_accuracy],
            'Precision': [male_precision, female_precision],
            'Recall': [male_recall, female_recall],
            'F1-score': [male_f1, female_f1]
        })
        display(metrics_table)

    return fpr_male, fpr_female, tpr_male, tpr_female


def print_male_female_metrics(model: any, X_: pd.DataFrame, X_male: pd.DataFrame, X_female: pd.DataFrame, X_test_: pd.DataFrame, y_test_: pd.Series) -> None:
    """
    Print the fairness metrics FPR and TPR for males and females, together with the statistical measures.
    :param model: the trained model
    :param X_: the original dataset without the target
    :param X_male: the original dataset without the target, only males
    :param X_female: the original dataset without the target, only females
    :param X_test_: the test set features
    :param y_test_: the test set target
    :return:
    """

    X_male_test, y_male_test, X_female_test, y_female_test = get_male_female_test_data(X_male, X_female, X_test_,
                                                                                       y_test_)
    split_testsets = [X_male_test, y_male_test, X_female_test, y_female_test]

    fpr_male, fpr_female, tpr_male, tpr_female = split_male_female_metrics(model, X_test_, y_test_,
                                                                           split_testsets=split_testsets)

    y_pred = model.predict(X_test_)
    X_test_with_sex = X_test_.copy()
    if 'sex' not in X_test_with_sex.columns:
        X_test_with_sex = X_test_.join(X_['sex'])
        X_test_with_sex['sex'] = X_test_with_sex['sex'].map({'Female': 0, 'Male': 1})

    DI, DS, EO, EOdds, conf_matrix = statistical_measures(X_test_with_sex, y_test_, y_pred, 'sex',
                                                          use_lib_implementation=False)

    print("Fairness Metrics:")
    print("Male FPR:", fpr_male)
    print("Male TPR:", tpr_male)
    print("Female FPR:", fpr_female)
    print("Female TPR:", tpr_female)
    print("")

    print_statistical_measures(DI, DS, EO, EOdds)


def calculate_composite_metric(accuracy: float, fpr_male: float, fpr_female: float, tpr_male: float, tpr_female: float,
                               accuracy_weight: float = 0.5, fairness_weight: float = 0.5):
    """
    Calculate the composite metric that balances accuracy and fairness.
    :param accuracy: Accuracy of the model.
    :param fpr_male: False Positive Rate (FPR) for males.
    :param fpr_female: False Positive Rate (FPR) for females.
    :param tpr_male: True Positive Rate (TPR) for males.
    :param tpr_female: True Positive Rate (TPR) for females.
    :param accuracy_weight: Weight assigned to accuracy (default: 0.7).
    :param fairness_weight: Weight assigned to fairness metrics (default: 0.3).
    :return: Composite metric value.
    """
    # Calculate fairness metric as the absolute difference between FPR for males and females
    fairness_metric = abs(fpr_male - fpr_female)

    # Calculate composite metric as a weighted sum of accuracy and fairness metrics
    composite_metric = accuracy_weight * accuracy + fairness_weight * (1 - fairness_metric)

    return composite_metric


def get_male_female_data(data: pd.DataFrame, is_encoded: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data into male-only data and female-only data.
    :param data: original dataset
    :param is_encoded: set to True if the dataset is already encoded (meaning 'sex' is already one-hot encoded)
    :return: male_data, female_data
    """
    if is_encoded:
        male_data = data[data['sex_Male'] == 1]
        female_data = data[data['sex_Male'] == 0]
        return male_data, female_data
    else:
        male_data = data[data['sex'] == 'Male']
        female_data = data[data['sex'] == 'Female']
        return male_data, female_data


def get_male_female_test_data(X_male, X_female, X_test_, y_test_) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Get test data for males and females. Returns Test data for males (features, target) and females (features, target).
    :param X_male: Dataframe containing male samples.
    :param X_female: Dataframe containing female samples.
    :param X_test_: Test features dataframe.
    :param y_test_: Test target series.
    :return: X_male_test, y_male_test, X_female_test, y_female_test
    """
    # Get the indices of male and female samples in the original dataset
    male_indices = X_male.index
    female_indices = X_female.index

    # Get the indices of male and female samples in the test set
    male_indices_test = X_test_.index.intersection(male_indices)
    female_indices_test = X_test_.index.intersection(female_indices)

    # Get the test data for males and females
    X_male_test = X_test_.loc[male_indices_test]
    y_male_test = y_test_.loc[male_indices_test]

    X_female_test = X_test_.loc[female_indices_test]
    y_female_test = y_test_.loc[female_indices_test]

    return X_male_test, y_male_test, X_female_test, y_female_test


def print_statistical_measures(DI, DS, EO, EOdds) -> None:
    """
    Print the fairness metrics.
    :param DI: Disparate Impact (DI) metric
    :param DS: Discrimination Score (DS) metric
    :param EO: Equal Opportunity (EO) metric
    :param EOdds: Equalized Odds (EOdds) metric
    :return:
    """
    print(f"Disparate Impact (DI): {DI:.3f}")
    print(f"Discrimination Score (DS): {DS:.3f}")
    print(f"Equal Opportunity Difference (EO): {EO:.3f}")
    print(f"Equalized Odds (EOdds): {EOdds:.3f}")


def print_conf_matrix(conf_matrix):
    """
    Print the FPR, TPR and PPP fairness metrics and the percentages of True Positives, True Negatives, False Positives, and False Negatives.
    :param conf_matrix: the confusion matrix, in our case, it is (needs to be) a 2x2 matrix
    :return:
    """
    total = conf_matrix.sum()
    tp = conf_matrix[1, 1]
    tn = conf_matrix[0, 0]
    fp = conf_matrix[0, 1]
    fn = conf_matrix[1, 0]

    print("\nConfusion Matrix: ")
    print("Percentage of True positives =", (tp / total))
    print("Percentage of True negatives =", (tn / total))
    print("Percentage of False positives =", (fp / total))
    print("Percentage of False negatives =", (fn / total))

    print("FPR: ", fp / (fp + tn))
    print("TPR: ", tp / (tp + fn))
    print("PPP: ", tp / (tp + fp))
