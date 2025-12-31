"""
===========================================================
Backblaze Hard Drive Failure Prediction
Model Training Script (XGBoost)
===========================================================

Goal:
- Train an industry-grade failure prediction model
- Handle extreme class imbalance correctly
- Avoid data leakage
- Produce metrics relevant to predictive maintenance

Why XGBoost?
- Strong performance on tabular data
- Handles non-linear interactions
- Robust to missing values
- Industry standard for predictive maintenance

IMPORTANT:
- Spark is used only for loading Parquet
- Pandas is used only after data is small enough
===========================================================
"""

# -------------------------------
# Imports
# -------------------------------
from pyspark.sql import SparkSession
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    classification_report
)

# -------------------------------
# Spark Session (Lightweight)
# -------------------------------
# We are NOT doing heavy Spark ops here
spark = (
    SparkSession.builder
    .appName("Backblaze-XGBoost-Training")
    .config("spark.driver.memory", "6g")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# -------------------------------
# Load Model-Ready Data
# -------------------------------
train_df = spark.read.parquet("data/model_ready/train.parquet")
val_df   = spark.read.parquet("data/model_ready/val.parquet")
test_df  = spark.read.parquet("data/model_ready/test.parquet")

print("Train rows:", train_df.count())
print("Val rows:", val_df.count())
print("Test rows:", test_df.count())

# -------------------------------
# Convert to Pandas (SAFE)
# -------------------------------
# These datasets are already downsampled
train_pd = train_df.toPandas()
val_pd   = val_df.toPandas()
test_pd  = test_df.toPandas()

# -------------------------------
# Feature / Label Split
# -------------------------------
TARGET = "failure_next_24h"

FEATURES = [
    "smart_5_raw",    # Reallocated Sectors
    "smart_187_raw",  # Uncorrectable Errors
    "smart_188_raw",  # Command Timeout
    "smart_197_raw",  # Pending Sectors
    "smart_198_raw",  # Offline Uncorrectable
    "smart_194_raw"   # Temperature
]

X_train = train_pd[FEATURES]
y_train = train_pd[TARGET]

X_val = val_pd[FEATURES]
y_val = val_pd[TARGET]

X_test = test_pd[FEATURES]
y_test = test_pd[TARGET]

# -------------------------------
# Handle Imbalance via Scale Pos Weight
# -------------------------------
# XGBoost prefers class weighting over naive oversampling
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

print(f"Scale_pos_weight: {pos_weight:.2f}")

# -------------------------------
# XGBoost Model Configuration
# -------------------------------
model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="aucpr",           # PR-AUC (critical for rare events)
    scale_pos_weight=pos_weight,   # Imbalance handling
    max_depth=6,
    n_estimators=500,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",            # Fast & memory efficient
    random_state=42
)

# -------------------------------
# Model Training with Early Stopping
# -------------------------------
model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=True,
    early_stopping_rounds=50
)

# -------------------------------
# Validation Evaluation
# -------------------------------
val_probs = model.predict_proba(X_val)[:, 1]
val_pr_auc = average_precision_score(y_val, val_probs)

print(f"\nValidation PR-AUC: {val_pr_auc:.4f}")

# -------------------------------
# Test Evaluation (Final)
# -------------------------------
test_probs = model.predict_proba(X_test)[:, 1]
test_pr_auc = average_precision_score(y_test, test_probs)

print(f"Test PR-AUC: {test_pr_auc:.4f}")

# -------------------------------
# Threshold Analysis (IMPORTANT)
# -------------------------------
# Default 0.5 is meaningless for rare-event prediction
precision, recall, thresholds = precision_recall_curve(y_test, test_probs)

# Example: choose threshold for ~80% recall
target_recall = 0.80
idx = next(i for i, r in enumerate(recall) if r >= target_recall)
chosen_threshold = thresholds[idx]

print(f"\nChosen threshold for {target_recall*100:.0f}% recall: {chosen_threshold:.4f}")

# -------------------------------
# Final Classification Report
# -------------------------------
test_preds = (test_probs >= chosen_threshold).astype(int)

print("\nFinal Test Classification Report:")
print(classification_report(y_test, test_preds, digits=4))

# -------------------------------
# Save Model
# -------------------------------
model.save_model("models/xgboost_backblaze.json")
print("Model saved to models/xgboost_backblaze.json")

spark.stop()
