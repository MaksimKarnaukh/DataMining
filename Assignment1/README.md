# DataMining

### Assignment 1: Association Rules

#### Project Structure for Assignment 1
```
Assignment1/
│
├── data/ (folder with the raw and cleaned dataset)
│   ├── assignment1_income_levels.xlsx
│   └── assignment1_income_levels_cleaned.csv
│
├── output/ (folder with the .txt files containing the association rules)
│   ├── a/ (everything related to Task 2.a)
│   └── b/ (everything related to Task 2.b)
│       └── split/ (related to the male/female dataset split)
│
├── plots/
│   ├── assoc_rules/ (plots related to Task 2, mostly heatmaps)
│   │   ├── a/ (plots related to Task 2.a) 
│   │   └── b/ (plots related to Task 2.b)
│   │       └── split/ (related to the male/female dataset split)
│   │  
│   └── data_preprocessing/ (plots related to Task 1)
│      
├── src/
│   ├── association_rules.ipynb (code for Task 2)
│   ├── data_preprocessing.ipynb (code for Task 1)
│   └── helper_functions.py
│
├── Data Mining Assignment - Association Rules.pdf
├── README.md
└── requirements.txt
```

The notebooks are commented and structured in a way that the code is easy to follow. 
The requirements.txt file contains the libraries used in the notebooks.