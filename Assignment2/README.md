# Data Mining

### Assignment 2: Classification

#### Project Structure for Assignment 2
```
Assignment2/
│
├── data/ (folder with the raw and cleaned dataset + test set without labels)
│   ├── assignment2_income_levels.xlsx
│   ├── assignment2_income_levels_cleaned.csv
│   └── assignment2_test.csv
│
├── output/ (folder with the saved models and predictions)
│   ├── saved_models/ 
│   └── pred/ 
│       
├── plots/
│   └── feat_importance_by_rules/ (folder with the heatmaps for 'income' association rules)
│       ├── high/ (folder with the plots for 'high' income consequent)
│       └── low/ (folder with the plots for 'low' income consequent)
│       
├── src/
│   ├── helper/ (folder with general helper stuff)
│   │   ├── clfmodel_functions.py (file with functions related to classification models)
│   │   ├── data_preprocessing.ipynb
│   │   ├── fairness_functions.py (file with functions related to fairness)
│   │   └── helper_functions.py (file with general helper functions)
│   │
│   ├── fairness.ipynb (some fairness experiments)
│   ├── feature_importance.ipynb (feature engineering and importance)
│   ├── model_decision_tree.ipynb
│   ├── model_knn.ipynb
│   ├── model_logistic_regression.ipynb
│   ├── model_naive_bayes.ipynb
│   └── model_random_forest.ipynb
│
├── Data Mining Assignment - Classification.pdf
├── README.md
└── requirements.txt
```

The notebooks are commented and structured in a way that the code is easy to follow. 
The requirements.txt file contains the libraries used in the notebooks.

This project explores the use of five different classification models to predict the income level of individuals based on a set of features.
In the files starting with `model_`, the models are trained and evaluated.

#### Sources and References

- Related to models:
  - https://www.kaggle.com/code/guruprasad91/titanic-naive-bayes
  - https://medium.com/analytics-vidhya/naive-bayes-for-mixed-typed-data-in-scikit-learn-fb6843e241f0
  - https://pypi.org/project/mixed-naive-bayes/#api-documentation
  - https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html#sphx-glr-auto-examples-model-selection-plot-multi-metric-evaluation-py
- Related to Feature Engineering:
  - https://www.geeksforgeeks.org/feature-selection-techniques-in-machine-learning/
  - https://www.analyticsvidhya.com/blog/2020/10/feature-selection-techniques-in-machine-learning/
- Related to Fairness:
  - [Developing a Novel Fair-Loan Classifier through a
      Multi-Sensitive Debiasing Pipeline: DualFair](https://cap.csail.mit.edu/sites/default/files/research-pdfs/make-04-00011-v2.pdf)
  - [Increasing fairness in supervised classification : a study of
simple decorrelation methods applied to the logistic regression (deSchaetzen)](https://dial.uclouvain.be/downloader/downloader.php?pid=thesis%3A30729&datastream=PDF_01&cover=cover-mem)
  - [Building Classifiers with Independency Constraints
](https://www.win.tue.nl/~mpechen/publications/pubs/CaldersICDM09.pdf)
  - [Fairness Constraints: Mechanisms for Fair Classification
](https://proceedings.mlr.press/v54/zafar17a/zafar17a.pdf)
  - [Auditing and Achieving Intersectional Fairness in Classification
Problems](https://arxiv.org/pdf/1911.01468.pdf)
  - [Classification with Fairness Constraints:
A Meta-Algorithm with Provable Guarantees](https://arxiv.org/pdf/1806.06055.pdf)
  - [Addressing Fairness in Classification with a Model-Agnostic Multi-Objective
Algorithm
](https://proceedings.mlr.press/v161/padh21a/padh21a.pdf)
  - https://medium.com/@janwithb/unmasking-bias-a-practical-example-of-bias-detection-for-a-loan-approval-ai-model-892ee3cb59f7
  - https://medium.com/north-east-data-science-review/diving-into-gender-bias-in-loan-approval-7472f288f77f
  - https://cs.yale.edu/bias/blog/jekyll/update/2018/11/06/fair-classification.html
  - https://jp-tok.dataplatform.cloud.ibm.com/docs/content/wsj/model/wos-fairness-metrics-ovr.html?context=cpdaas
  - https://machinelearningmastery.com/random-oversampling-and-undersampling-for-imbalanced-classification/
