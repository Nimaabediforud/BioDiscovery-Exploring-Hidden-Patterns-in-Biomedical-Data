# BioDiscovery: Exploring Hidden Patterns in Biomedical Data

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-1.x-blue?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-success?logo=pandas)
![Status](https://img.shields.io/badge/Status-Completed-success?style=flat-square)


An end-to-end unsupervised machine learning framework for biomedical data exploration using thyroid cancer clinical data. This project investigates hidden patient patterns through clustering, dimensionality reduction, anomaly detection, and association rule mining while emphasizing reproducible preprocessing pipelines, benchmarking, and visualization.

---

## Project Objectives

This project aims to:

- Explore hidden structures within biomedical datasets using unsupervised learning.
- Compare multiple clustering algorithms through a unified benchmarking framework.
- Visualize high-dimensional biomedical data using PCA, t-SNE, and UMAP.
- Detect abnormal patient profiles using anomaly detection algorithms.
- Discover clinically meaningful feature relationships through association rule mining.
- Build reusable preprocessing pipelines for different unsupervised learning paradigms.
- Provide an educational and reproducible workflow for biomedical data analysis.

---

## Dataset

The project uses the **Thyroid Cancer Recurrence Dataset**, containing clinicopathologic characteristics of thyroid cancer patients.

The dataset includes demographic, pathological, TNM staging, treatment response, and recurrence information.

Throughout the project, the recurrence variable is excluded from unsupervised model training and reserved for external validation and post hoc interpretation whenever appropriate.

Source:
https://www.kaggle.com/datasets/jainaru/thyroid-disease-data

---

# Repository Structure

```
BioDiscovery/
│
├── README.md
├── LICENSE
├── .gitignore
│
├── Data/
│   ├── Original dataset
│   └── Processed datasets
|
├── Models/
│   ├── Best clustering model
│   └── Best anomaly detection model
│   
│
└── Notebooks/
    ├── EDA.ipynb
    ├── preprocessing.ipynb
    ├── clustering_analysis.ipynb
    ├── dimensionality_reduction.ipynb
    ├── anomaly_detection.ipynb
    ├── association_rule_mining.ipynb
    ├── representation_learning.ipynb
    └── utils.py
```

---

# Project Workflow

```
Raw Dataset
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Preprocessing Pipelines
      │
      ├───────────────┐
      │               │
      ▼               ▼
 Feature Set A    Feature Set B
      │               │
      ├───────────────┘
      │
      ▼
Unsupervised Learning
      │
      ├── Clustering
      ├── Dimensionality Reduction
      ├── Anomaly Detection
      ├── Representation Learning
      └── Association Rule Mining
      │
      ▼
Visualization & Interpretation
```

---

# Methods

## Exploratory Data Analysis

- Missing value analysis
- Duplicate inspection
- Feature distributions
- Correlation analysis
- Outlier inspection
- Clinical variable exploration

---

## Preprocessing

Separate preprocessing pipelines were developed for different unsupervised learning tasks.

Techniques include:

- Missing value imputation
- One-hot encoding
- Ordinal encoding
- Standardization
- PCA transformation
- Transaction dataset construction for association rule mining

---

## Clustering Analysis

Implemented algorithms:

- K-Means
- Agglomerative Clustering
- DBSCAN
- Gaussian Mixture Models

Evaluation metrics:

- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index
- Adjusted Rand Index (ARI) 
- Normalized Mutual Information (NMI)

---

## Dimensionality Reduction & Representation Learning

Methods:

- Principal Component Analysis (PCA)
- t-SNE
- UMAP

PCA was used for linear dimensionality reduction and for generating reduced feature spaces (80% explained variance) used in subsequent analyses.

t-SNE and UMAP were investigated separately as representation learning techniques to visualize hidden structures and compare embeddings generated from:

- Original feature space
- PCA-transformed feature space (80% explained variance)

---

## Anomaly Detection

Algorithms:

- Isolation Forest
- Local Outlier Factor (LOF)
- One-Class SVM

Evaluation included:

- Precision
- Recall
- F1-score
- Confusion Matrix
- Anomaly score distributions
- PCA-based anomaly visualization

---

## Association Rule Mining

Algorithms:

- Apriori
- FP-Growth

Rule evaluation metrics:

- Support
- Confidence
- Lift
- Leverage
- Conviction
- Jaccard
- Kulczynski
- Zhang's Metric
- Certainty Factor
- Representativity

Visualization techniques include:

- Metric distributions
- Support vs Confidence
- Lift analysis
- Metric correlation heatmaps

---

# Key Results

- Performed comprehensive exploratory data analysis of thyroid cancer clinicopathologic data.
- Developed a complete preprocessing pipeline with multiple feature representations.
- Identified meaningful patient subgroups using multiple clustering algorithms.
- Reduced feature dimensionality using Principal Component Analysis (PCA).
- Compared nonlinear representations using t-SNE and UMAP, demonstrating differences in local and global structure preservation.
- Detected anomalous patient profiles using Isolation Forest, Local Outlier Factor, and One-Class SVM.
- Discovered clinically meaningful associations among clinicopathologic variables using Apriori and FP-Growth algorithms.
- Benchmarked multiple unsupervised learning algorithms using quantitative evaluation metrics and visualization techniques.

---

# Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- mlxtend
- UMAP
- Matplotlib
- Seaborn
- Joblib
- Jupyter Notebook

---

## License

This project is released under the MIT License.