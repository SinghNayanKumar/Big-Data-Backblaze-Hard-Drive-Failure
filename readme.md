
# 🧠 Backblaze Hard Drive Failure Prediction

### End-to-End Big-Data ML System for Rare Event Prediction

---

## 🔍 Project Overview

This project implements a **production-style failure prediction system** for hard drives using Backblaze SMART telemetry data.

The objective is not just to train a model, but to **design, evaluate, and stress-test an end-to-end ML system** under real-world constraints:

* Millions of rows of time-series data
* Extreme class imbalance (≈ 1 failure in 20,000 samples)
* Strict temporal integrity (no data leakage)
* Operationally meaningful evaluation beyond PR-AUC

The result is a **Spark-first, deployment-oriented pipeline** that demonstrates how predictive maintenance systems behave in practice — including their limitations.

---

## 🎯 Problem Statement

Predict whether a hard drive will fail **within the next 24 hours**, using SMART attributes recorded daily.

This is a **rare-event prediction problem** with severe asymmetry in cost:

* Missing a failure → potential data loss
* Excessive false alerts → operational overload

---

## 🧩 Core Challenges Addressed

* **Extreme class imbalance** (≈ 0.005% positives)
* **Time-dependent labels** (future failures)
* **Scalability** to millions of rows
* **Mismatch between model metrics and operational impact**
* **Explainability under noisy, sparse features**

---

## 🏗️ System Architecture

```
Raw SMART Logs (Parquet)
        ↓
Spark-based Feature Processing
        ↓
Time-aware Label Generation
        ↓
Train / Validation / Test Split (chronological)
        ↓
Downsampled Training (Spark → Pandas)
        ↓
XGBoost Model Training
        ↓
Full-scale Spark Inference
        ↓
Operational Impact Simulation
        ↓
SHAP-based Alert Explanation
```

---

## 📁 Repository Structure

```
.
├── data/
│   ├── raw/                 # Original Backblaze SMART logs
│   ├── processed/           # Feature-engineered datasets
│   └── model_ready/         # Time-based train/val/test splits
│
├── models/
│   └── xgboost_backblaze.json
│
├── outputs/
│   ├── daily_alerts.parquet
│   └── shap_alert_summary.png
│
├── scripts/
│   ├── 01_data_processing.py
│   ├── 02_label_generation.py
│   ├── 03_train_val_test_split.py
│   ├── 04_model_training.py
│   ├── 05_SHAP_alert_explanation.py
│   └── 06_deployment_code.py
│
└── README.md
```

---

## 🔄 Data Splitting Strategy (No Leakage)

To preserve temporal causality, the dataset is split **strictly by time**:

| Split      | Purpose                                |
| ---------- | -------------------------------------- |
| Train      | Model fitting                          |
| Validation | Hyperparameter tuning / early stopping |
| Test       | Future unseen data                     |

Random splits are **explicitly avoided**.

---

## ⚖️ Handling Extreme Imbalance

* Raw class ratio ≈ **1 : 25,000**
* Strategy:

  * Negative class downsampling **only in training**
  * `scale_pos_weight` used in XGBoost
* Validation and test sets remain **fully imbalanced** to reflect reality

This ensures:

* Learnable signal during training
* Honest evaluation during inference

---

## 🤖 Model Choice

**XGBoost (Gradient Boosted Trees)**

Why:

* Strong performance on tabular data
* Handles non-linear feature interactions
* Robust to missing values
* Widely used in industrial reliability systems

**Evaluation metric:** PR-AUC
(ROC-AUC is misleading under extreme imbalance)

---

## 🚀 Deployment-Style Inference

* Full test set scored using **Spark**
* XGBoost inference applied via Spark UDF
* Produces per-drive, per-day failure probabilities

This simulates **real batch deployment**, not offline evaluation.

---

## 📊 Operational Impact Analysis

Instead of stopping at PR-AUC, the system is evaluated in operational terms:

* Detected failures (TP)
* Missed failures (FN)
* False alarms (FP)
* Alerts per day

### Example Outcome

```
Total test failures:        248
Detected failures (TP):     202
Missed failures (FN):       46
False alarms (FP):          2,525,188

Recall:                     0.81
Precision:                  0.0001
Avg alerts per day:         157,836
```

🔎 **Key Insight**
A model can achieve high recall yet be **operationally unusable** due to alert fatigue — a common failure mode in rare-event ML systems.

---

## 🔍 Model Interpretability (SHAP)

SHAP analysis is applied to alerted samples to understand model behavior.

Findings:

* Model correctly emphasizes SMART attributes linked to disk degradation
* Sparse counters and noisy signals cause large SHAP variance
* Explains alert explosion despite reasonable recall

This reinforces the need for **policy-based alerting**, not naive probability thresholds.

---

## 🧠 Key Takeaways

* Rare-event prediction is a **systems problem**, not just a modeling problem
* PR-AUC alone is insufficient for deployment decisions
* Threshold-based alerting breaks down under extreme imbalance
* Spark-first pipelines are mandatory at this scale
* Explainability is critical for diagnosing failure modes

---

## 🚧 Known Limitations (Intentional)

* No probability calibration
* No ranking-based alert budget (Top-K/day)
* No online learning loop

These are consciously excluded to keep the project focused on **core system design principles**.

---

## 🏁 Project Status

**Complete.**
This repository is intended as a **capability demonstration** for:

* Big-data ML engineering
* Failure prediction systems
* Imbalanced learning
* Deployment-oriented evaluation

---

## 👤 Author

**Nayan**
Interests:

* Large-scale ML systems
* Reliability & predictive maintenance
* AI for operational decision-making


