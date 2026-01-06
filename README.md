# Business Intelligence Project

## Faculty of Computers and Artificial Intelligence – Helwan University

**Subject:** Business Intelligence
**Student:** Bassam
**Program:** Computer Science & Artificial Intelligence

---

## 📌 Project Overview

This project is part of the **Business Intelligence** course at Helwan University. The main objective of the project is to apply **data mining and machine learning techniques** on different datasets, build **multiple models**, and **compare their performance** to extract meaningful business insights.

Each notebook focuses on a specific BI task such as **classification, regression, clustering, and association rule mining**, where **two different models/algorithms** are implemented and compared.

---

## 📂 Project Structure

```
├── Apriori_FP_Growth_Groceries.ipynb
├── Classification_models.ipynb
├── Cluster_models.ipynb
├── Linear_Reg_implement.ipynb
├── Regression_models.ipynb
├── Customers.csv
├── Groceries_dataset.csv
├── Regression dataset.csv
├── classification dataset.csv
├── studytime_score.csv
└── README.md
```

---

## 📘 Notebooks Description

### 1️⃣ Apriori & FP-Growth (Association Rule Mining)

**File:** `Apriori_FP_Growth_Groceries.ipynb`

* Dataset: `Groceries_dataset.csv`
* Models Used:

  * Apriori Algorithm
  * FP-Growth Algorithm
* Objective:

  * Discover frequent itemsets and association rules.
  * Compare execution time, number of rules, and efficiency.

---

### 2️⃣ Classification Models

**File:** `Classification_models.ipynb`

* Dataset: `classification dataset.csv`
* Models Used:

  * K-Nearest Neighbors (KNN)
  * Naive Bayes
* Evaluation Metrics:

  * Accuracy
  * Confusion Matrix
  * Precision, Recall, F1-score
* Goal:

  * Compare instance-based learning (KNN) with probabilistic classification (Naive Bayes).
  * Analyze performance differences and suitability for the dataset.

---

### 3️⃣ Clustering Models

**File:** `Cluster_models.ipynb`

* Dataset: `Customers.csv`
* Models Used:

  * K-Means Clustering
  * Gaussian Mixture Model (GMM)
* Objective:

  * Cluster customers into meaningful segments.
  * Compare hard clustering (K-Means) with probabilistic clustering (GMM).
  * Interpret clusters from a business perspective.

---

### 4️⃣ Regression Models

#### 🔹 Linear Regression Implementation

**File:** `Linear_Reg_implement.ipynb`

* Dataset: `studytime_score.csv`
* Focus:

  * Implement Linear Regression.
  * Understand the relationship between independent and dependent variables.

#### 🔹 Regression Model Comparison

**File:** `Regression_models.ipynb`

* Dataset: `Regression dataset.csv`
* Models Used:

  * Linear Regression
  * Random Forest Regression
* Evaluation Metrics:

  * Mean Squared Error (MSE)
  * R² Score
* Goal:

  * Compare a simple parametric model with an ensemble-based non-linear model.

---

## 🔍 Model Comparison Strategy

For each notebook:

* Two different models are implemented.
* Performance is evaluated using appropriate metrics.
* Results are compared to identify the best-performing model.
* Business insights are extracted from the results.

---

## 🛠️ Tools & Technologies

* Python
* Jupyter Notebook
* Libraries:

  * NumPy
  * Pandas
  * Matplotlib / Seaborn
  * Scikit-learn
  * Mlxtend (for Apriori & FP-Growth)

---

## 🎯 Learning Outcomes

* Apply Business Intelligence concepts practically.
* Understand strengths and weaknesses of different models.
* Perform data preprocessing, modeling, and evaluation.
* Gain experience in model comparison and result interpretation.

---

## 📌 Conclusion

This project demonstrates how **Business Intelligence techniques** can be used to analyze data, build predictive models, and support decision-making. Comparing multiple models helps in selecting the most suitable approach for different business problems.

---

## 📎 Notes

This project is developed for **academic purposes** as part of the Business Intelligence course at Helwan University.
