# 💳 Credit Card Fraud Detection System

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-black?style=for-the-badge&logo=xgboost)

A comprehensive data science project focused on identifying fraudulent credit card transactions. This repository covers the entire pipeline from raw data merging and exploratory analysis to advanced model ensembling and the development of a multipage GUI application.

## 🚀 Project Overview

The primary objective of this project is to develop a robust machine learning system capable of detecting fraudulent transactions in a highly imbalanced dataset (only ~1.6% fraud). We prioritize **Recall** to ensure that as many fraudulent transactions as possible are caught, while maintaining a reasonable **Precision** floor to minimize false alarms.

### Key Milestones:
- **Data Integration**: Merging disparate datasets using unique identifiers.
- **Deep EDA**: Comprehensive statistical analysis and visualization of 19+ features.
- **Advanced Feature Engineering**: Creating domain-specific features like `INCOME_PER_PERSON` and `TENURE_RATIO`.
- **Model Ensembling**: Implementing Balanced Random Forests, XGBoost, and Stacking Classifiers.
- **Interactive GUI**: A multipage application for data exploration and real-time fraud prediction.

---

## 📊 Dataset Description

The analysis is performed on a credit card dataset containing **25,134 transactions** with features including:
- **Demographics**: Gender, Family Type, Education, Age.
- **Financials**: Income, Income Type, Employment History.
- **Target**: Binary classification (0: Legitimate, 1: Fraudulent).

---

## 🛠️ Tech Stack

- **Data Processing**: `Pandas`, `NumPy`, `Scipy`
- **Machine Learning**: `Scikit-Learn`, `XGBoost`, `Imbalanced-Learn (SMOTE)`
- **Visualization**: `Matplotlib`, `Seaborn`
- **Model Persistence**: `Joblib`

---

## 📂 Project Structure

```text
├── archive/                 # Raw datasets (.csv)
├── exports/                 # Visualizations and serialized models (.svg, .joblib)
├── notebooks/               # Jupyter notebooks for EDA and modeling iterations
│   ├── exploring.ipynb      # Initial data exploration
│   ├── modelling_v1.ipynb   # Baseline models
│   └── modelling_v2.ipynb   # Advanced pipeline & stacking
└── README.md                # Project documentation
```

---

## 📈 Model Performance & Improvements

### Current Best Model: **Stacking Ensemble**
We utilize a Stacking Classifier with **Balanced Random Forest** and **Weighted XGBoost** as base learners, and **Logistic Regression** as the meta-learner.

| Improvement Strategy | Impact |
| :--- | :--- |
| **Feature Engineering** | 15% increase in feature importance for engineered variables. |
| **Recall-Oriented Scoring** | Optimized models specifically for `Recall @ Precision ≥ 10%`. |
| **Smart Thresholding** | Dynamic probability threshold selection via PR-curves. |
| **Class Imbalance Handling** | SMOTE and Weighted Loss functions to handle sparse fraud cases. |

---

## ⚙️ Getting Started

### Prerequisites
- Python 3.8+
- Recommended: `pip` or `conda` environment

### Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Credit
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Analysis
- Explore the EDA and modeling logic in `notebooks/`.

---

