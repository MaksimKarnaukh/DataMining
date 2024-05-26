# Data Mining

### Assignment 3: Clustering

#### Project Structure for Assignment 3
```
Assignment1/
│
├── data/ (folder with articles dataset)
│   └── assignment3_articles.csv
│
├── output/
│   ├── cluster_assignments/ (original dataset file with cluster assignments)
│   └── plots/ (plots for the paper)
│      
├── src/
│   ├── cluster_plotting.py (cluster plotting functions)
│   ├── cluster_validation.py (cluster validation functions)
│   ├── cluster_validation_testing.ipynb (cluster validation testing on small example 2D dataset)
│   ├── clustering_agglomerative.ipynb
│   ├── clustering_bisectingkmeans.ipynb
│   ├── clustering_dbscan.ipynb (this file shouldn't be looked at too seriously)
│   ├── clustering_kmeans.ipynb
│   ├── clustering_kmedoids.ipynb
│   ├── data_preprocessor.py (data preprocessor class and functions)
│   └── helper_functions.py
│
├── Data Mining Assignment - Clustering.pdf
├── README.md
└── requirements.txt
```

The notebooks are commented and structured in a way that the code is easy to follow. 
The requirements.txt file contains the libraries used in the notebooks.

Please use a python 3.11 (or below) environment to run the code. 
Python 3.12 doesn't work well with the sklearn_extra.cluster library (problem with distutils import in the library itself).