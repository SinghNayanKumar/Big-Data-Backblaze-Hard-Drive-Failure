"""
===========================================================
Backblaze Hard Drive Failure Prediction
Deployment & Operational Impact (Spark-first)
===========================================================

PURPOSE
-------
This script simulates a REAL production deployment:

- Batch inference on future (test) data
- Threshold-based alerting
- Spark-first execution (no Pandas on large data)
- Operationally meaningful metrics:
    - Recall
    - Precision
    - Alerts per day
    - Missed failures

IMPORTANT DESIGN CHOICES
------------------------
- XGBoost is used only for inference
- Spark handles all large-scale data
- Threshold is external decision policy (NOT in model)
- Feature-name alignment is enforced (CRITICAL FIX)

This script answers:
"Would operations actually accept this model?"
===========================================================
"""

# ===========================================================
# Imports
# ===========================================================
import os
import json
import numpy as np
import xgboost as xgb

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ===========================================================
# Spark Session (Read-heavy, inference-safe)
# ===========================================================
spark = (
    SparkSession.builder
    .appName("Backblaze-Deployment-Operational-Impact")
    .config("spark.driver.memory", "8g")
    .config("spark.executor.memory", "8g")
    .config("spark.sql.shuffle.partitions", "200")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# ===========================================================
# Load FULL Test Dataset (Future / Unseen)
# ===========================================================
test_df = spark.read.parquet(
    "data/model_ready/test.parquet"
)

print("Test rows:", test_df.count())
print(
    "Test positive rows:",
    test_df.filter(F.col("failure_next_24h") == 1).count()
)

# ===========================================================
# Load Trained XGBoost Model
# ===========================================================
model = xgb.Booster()
model.load_model("models/xgboost_backblaze.json")

# ===========================================================
# Load Threshold Metadata
# ===========================================================
with open("models/threshold.json") as f:
    threshold_meta = json.load(f)

THRESHOLD = threshold_meta["threshold"]
TARGET_RECALL = threshold_meta["target_recall"]

print(f"Using threshold: {THRESHOLD:.6f}")
print(f"Target recall: {TARGET_RECALL:.2f}")

# ===========================================================
# Feature Schema (MUST match training exactly)
# ===========================================================
FEATURES = [
    "smart_5_raw",     # Reallocated sectors
    "smart_187_raw",   # Uncorrectable errors
    "smart_188_raw",   # Command timeout
    "smart_197_raw",   # Pending sectors
    "smart_198_raw",   # Offline uncorrectable
    "smart_194_raw"    # Temperature
]

TARGET = "failure_next_24h"

# ===========================================================
# Spark → XGBoost Inference UDF (FIXED)
# ===========================================================
# CRITICAL FIX:
# - XGBoost was trained with feature names
# - We MUST pass feature_names into DMatrix
# - Otherwise Spark jobs crash at runtime

def predict_proba_udf(*cols):
    """
    Convert a Spark row into an XGBoost probability prediction.
    This function runs on Spark executors.
    """
    X = np.array(cols, dtype=float).reshape(1, -1)

    dmatrix = xgb.DMatrix(
        X,
        feature_names=FEATURES  # <<< FIX: enforce feature alignment
    )

    return float(model.predict(dmatrix)[0])

predict_udf = F.udf(predict_proba_udf)

# ===========================================================
# Batch Inference (Spark-first)
# ===========================================================
scored_df = test_df.withColumn(
    "failure_probability",
    predict_udf(*FEATURES)
)

# ===========================================================
# Apply Decision Threshold (Deployment Logic)
# ===========================================================
scored_df = scored_df.withColumn(
    "alert",
    F.when(F.col("failure_probability") >= THRESHOLD, 1).otherwise(0)
)

# ===========================================================
# Core Operational Metrics
# ===========================================================

# Total future failures
total_failures = scored_df.filter(F.col(TARGET) == 1).count()

# True positives (detected failures)
true_positives = scored_df.filter(
    (F.col("alert") == 1) & (F.col(TARGET) == 1)
).count()

# False positives (false alarms)
false_positives = scored_df.filter(
    (F.col("alert") == 1) & (F.col(TARGET) == 0)
).count()

# False negatives (missed failures)
false_negatives = scored_df.filter(
    (F.col("alert") == 0) & (F.col(TARGET) == 1)
).count()

# ===========================================================
# Derived Metrics (Ops-relevant)
# ===========================================================
recall = (
    true_positives / total_failures
    if total_failures > 0 else 0.0
)

precision = (
    true_positives / (true_positives + false_positives)
    if (true_positives + false_positives) > 0 else 0.0
)

# Alerts per day (alert fatigue metric)
alerts_per_day = (
    scored_df
    .groupBy("date")
    .agg(F.sum("alert").alias("alerts"))
    .agg(F.avg("alerts"))
    .collect()[0][0]
)

# ===========================================================
# Print Operational Summary
# ===========================================================
print("\n================ OPERATIONAL IMPACT =================")
print(f"Total test failures:        {total_failures}")
print(f"Detected failures (TP):     {true_positives}")
print(f"Missed failures (FN):       {false_negatives}")
print(f"False alarms (FP):          {false_positives}")
print("----------------------------------------------------")
print(f"Recall:                     {recall:.4f}")
print(f"Precision:                  {precision:.4f}")
print(f"Avg alerts per day:         {alerts_per_day:.1f}")
print("====================================================")

# ===========================================================
# Persist Alerts for Ops / Downstream Systems
# ===========================================================
os.makedirs("outputs", exist_ok=True)

alerts_df = scored_df.filter(F.col("alert") == 1)

alerts_df.write.mode("overwrite").parquet(
    "outputs/daily_alerts.parquet"
)

print("Alert dataset written to outputs/daily_alerts.parquet")

spark.stop()
