"""
===========================================================
Backblaze Hard Drive Failure Prediction
Model Training Script (XGBoost)
===========================================================

Goal:
- Train an industry-grade predictive maintenance model
- Correctly handle extreme class imbalance
- Prevent temporal leakage
- Avoid memory crashes (OOM-safe)
- Produce PR-AUC and threshold-based metrics

Why XGBoost?
- Strong performance on tabular data
- Handles non-linear interactions
- Robust to missing values
- Industry standard for predictive maintenance

CRITICAL DESIGN CHOICES:
- Spark is used for ALL large datasets (val/test)
- Pandas is used ONLY for the downsampled training set
- Model inference & evaluation on val/test stays in Spark

This mirrors how Databricks / Azure ML / AWS EMR pipelines
are built in real production systems.
===========================================================
"""

# -------------------------------
# Imports
# -------------------------------
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml.evaluation import BinaryClassificationEvaluator

import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.metrics import average_precision_score

# -------------------------------
# Spark Session (Lightweight & Safe)
# -------------------------------
# We do NOT perform heavy Spark transformations here.
# Spark is only used for:
# - Loading Parquet
# - Scoring large datasets
spark = (
    SparkSession.builder
    .appName("Backblaze-XGBoost-Training")
    .config("spark.driver.memory", "6g")
    .config("spark.sql.shuffle.partitions", "200")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# -------------------------------
# Load Model-Ready Datasets
# -------------------------------
train_df = spark.read.parquet("data/model_ready/train.parquet")
val_df   = spark.read.parquet("data/model_ready/val.parquet")
test_df  = spark.read.parquet("data/model_ready/test.parquet")

print("Train rows:", train_df.count())
print("Val rows:", val_df.count())
print("Test rows:", test_df.count())

# -------------------------------
# Convert ONLY TRAINING DATA to Pandas (SAFE)
# -------------------------------
# Training data was explicitly downsampled earlier.
# This is the ONLY dataset allowed into pandas.
train_pd = train_df.toPandas()

# -------------------------------
# Feature / Label Definition
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

# -------------------------------
# Handle Extreme Class Imbalance
# -------------------------------
# XGBoost performs best with class weighting
# rather than naive upsampling.
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

print(f"Scale_pos_weight: {pos_weight:.2f}")

# -------------------------------
# XGBoost Model Configuration
# -------------------------------
model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="aucpr",            # PR-AUC is correct for rare failures
    scale_pos_weight=pos_weight,    # Critical for imbalance
    max_depth=6,
    n_estimators=500,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",             # Fast + memory efficient
    random_state=42
)

# -------------------------------
# Train Model (No Leakage)
# -------------------------------
model.fit(X_train, y_train, verbose = True)

print("Model training completed")

# -------------------------------
# Spark-Safe Inference Function
# -------------------------------
# We broadcast the trained model and score rows
# WITHOUT converting Spark → Pandas
def predict_proba_udf(*cols):
    X = np.array(cols, dtype=float).reshape(1, -1)
    return float(model.predict_proba(X)[0, 1])

predict_udf = F.udf(predict_proba_udf, DoubleType())

# -------------------------------
# Score Validation Set (Spark)
# -------------------------------
val_scored = val_df.withColumn(
    "failure_probability",
    predict_udf(*FEATURES)
)

# -------------------------------
# Score Test Set (Spark)
# -------------------------------
test_scored = test_df.withColumn(
    "failure_probability",
    predict_udf(*FEATURES)
)

# -------------------------------
# Evaluation (Spark-Native)
# -------------------------------
# PR-AUC is the correct metric for predictive maintenance
evaluator = BinaryClassificationEvaluator(
    labelCol=TARGET,
    rawPredictionCol="failure_probability",
    metricName="areaUnderPR"
)

val_pr_auc = evaluator.evaluate(val_scored)
#test_pr_auc = evaluator.evaluate(test_scored)

print(f"\nValidation PR-AUC: {val_pr_auc:.4f}")
#print(f"Test PR-AUC: {test_pr_auc:.4f}")

# -------------------------------
# Threshold Selection (Business-Aware)
# -------------------------------
# Convert a SMALL subset of test predictions for threshold tuning
# (safe, controlled, intentional)
threshold_pd = (
    val_scored
    .select(TARGET, "failure_probability")
    .sample(fraction=0.05, seed=42)
    .toPandas()
)



from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(
    threshold_pd[TARGET],
    threshold_pd["failure_probability"]
)


# Choose threshold for high recall (failure detection priority)
target_recall = 0.80
idx = next(i for i, r in enumerate(recall) if r >= target_recall)
chosen_threshold = thresholds[idx]

print(f"\nChosen threshold for {target_recall*100:.0f}% recall: {chosen_threshold:.4f}")

# -------------------------------
# Save Model
# -------------------------------
model.get_booster().save_model("models/xgboost_backblaze.json")
print("Model saved to models/xgboost_backblaze.json")

#Save Threshold Metadata
import json

threshold_metadata = {
    "threshold": float(chosen_threshold),
    "target_recall": target_recall,
    "metric": "PR-AUC",
    "notes": "Chosen for high-recall failure detection"
}

with open("models/threshold.json", "w") as f:
    json.dump(threshold_metadata, f, indent=2)


spark.stop()
