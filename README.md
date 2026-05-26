# Fraud Detection System (ML + FastAPI)

An end-to-end Machine Learning project for detecting fraudulent credit card transactions using anomaly detection and classification models, deployed via FastAPI.

---

## Project Overview

This system analyzes credit card transactions and predicts whether a transaction is:

- Non-Fraud
- Fraud

It also provides:
- Fraud probability score
- Risk level classification (Low / Medium / High)

---

## Machine Learning Pipeline

### 1. Data Processing
- Handling missing values
- Removing duplicates
- Feature scaling (StandardScaler)
- Log transformation for Amount
- Train/Test split

### 2. Imbalanced Data Handling
- SMOTE (Synthetic Minority Oversampling Technique)

### 3. Models Used
- Logistic Regression
- Random Forest
- XGBoost

### 4. Evaluation Metrics
- Accuracy
- Precision
- Recall (very important for fraud detection)
- F1-score
- Confusion Matrix

---

## Key Insight

- Dataset is highly imbalanced (~0.17% fraud cases)
- Recall is prioritized over accuracy to reduce False Negatives
- Feature importance shows strong signals in V14, V10, V12, V17

---

## Features

Input features used by the model:

- Time
- V1 → V28 (PCA transformed features)
- Amount_Log

---

## Model Output

The API returns:

json: 
{
  "prediction": "Fraud / Non-Fraud",
  "fraud_probability": 0.87,
  "risk_level": "Low / Medium / High"
}
API Endpoints
Home
GET /
Prediction
POST /predict
Example Request
{
  "Time": 0.5,
  "V1": -1.2,
  "V2": 0.3,
  "V3": -0.8,
  ...
  "V28": 0.1,
  "Amount_Log": 3.2
}
Example Response
{
  "prediction": "Fraud",
  "fraud_probability": 0.91,
  "risk_level": "High"
}

---

## 🧠 Machine Learning Models & Selection

experimenting with multiple machine learning models to solve the fraud detection problem:

### Models Tested:
- Logistic Regression
- Random Forest
- XGBoost 

---

## Best Model Selection

After extensive evaluation and comparison using multiple metrics (Precision, Recall, F1-score, and Confusion Matrix), I found that:
Random Forest performed the best overall,
Key reasons:
- Highest generalization performance on unseen data
- Better balance between Precision and Recall
- More stable compared to other models
- Strong performance on highly imbalanced dataset

---

## Why Recall Matters Here

In fraud detection, minimizing False Negatives is critical:

- Missing fraud (False Negative) is more costly than false alerts
- Therefore, Recall was prioritized over Accuracy

---

## Final Decision

Random Forest was selected as the final production model due to its superior balance between performance, stability, and real-world applicability.

---

## Tech Stack
Python
Pandas, NumPy
Scikit-learn
imbalanced-learn (SMOTE)
XGBoost
FastAPI
Uvicorn

--- 

How to Run
1. Install dependencies
pip install -r requirements.txt
2. Run API
uvicorn app:app --reload

---