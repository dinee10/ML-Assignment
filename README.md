# ML-Assignment: Heart Failure Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Project Overview
This project applies classical machine learning algorithms to predict **heart failure outcomes** (survival vs. death event) based on clinical records of patients. The goal is to compare multiple machine learning models and evaluate their performance on the same dataset.

## Problem Statement
Heart failure is a major global health concern. Early and accurate prediction of patient outcomes can help clinicians make better decisions and improve survival rates. This project develops machine learning models that classify patients as likely to experience a **death event** (1) or survive (0) during the follow-up period using 12 clinical features.

## Dataset
The dataset used in this project is the **Heart Failure Clinical Records** dataset available on the UCI Machine Learning Repository.

**Dataset Link:** [https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records)

- **Number of Instances:** 299 patients
- **Number of Features:** 13 (12 clinical features + 1 target)
- **Data Format:** CSV (`heart_failure_clinical_records_dataset.csv`)

## Machine Learning Algorithms Used
Four different machine learning algorithms are implemented and compared:

- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **K-Means Clustering**
- **Random Forest**

Each group member is responsible for implementing and evaluating one algorithm.

> **Important Note on K-Means:**  
> K-Means is an **unsupervised** clustering algorithm, while the other three are **supervised** classification algorithms. In this project, K-Means is used to discover natural patient groupings based on clinical features, and its clusters are later compared against the true `DEATH_EVENT` labels for evaluation.

## Data Preprocessing
The following preprocessing steps were applied to all models:

- **Handling missing values:** None present (dataset is clean)
- **Feature scaling:** `StandardScaler` (required for KNN, Logistic Regression, and K-Means)
- **Train-test split:** 80% training / 20% testing (for supervised models)
- **For K-Means:** Applied on the full feature set (or scaled features) without target leakage

## How to Compare These 4 Models
Since three models are supervised classifiers and one is unsupervised, we use a **two-tier evaluation approach**:

### 1. Supervised Models (Logistic Regression, KNN, Random Forest)
We evaluate using standard classification metrics on the **test set**:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **Confusion Matrix**
- **ROC-AUC** (optional for better comparison)

### 2. K-Means (Unsupervised)
- Use **Elbow Method** + **Silhouette Score** to determine optimal number of clusters (usually `k=2` to match binary target).
- After clustering, map the two clusters to `DEATH_EVENT` labels (majority vote) and compute the same classification metrics above for fair comparison.
- Additional metrics: Inertia (within-cluster sum of squares), Silhouette Score.



## Repository Structure
