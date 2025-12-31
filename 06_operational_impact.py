"""
===========================================================
Backblaze Hard Drive Failure Prediction
Operational Impact Simulation (Spark-first)
===========================================================

GOAL
----
Translate model predictions into *operational reality*:
- How many alerts per day?
- What precision & recall do we achieve?
- How many failures do we still miss?

WHY THIS MATTERS
----------------
PR-AUC alone does NOT tell us:
- Alert fatigue
- Ops cost
- SLA violations

This script simulates a real deployment:
- Daily batch inference
- Threshold-based alerting
- Metrics meaningful to operations teams

IMPORTANT
---------
- Spark is used end-to-end (no Pandas conversion)
- Model outputs probabilities
- Threshold is applied as a *decision policy*
===========================================================
"""

# -------------------------------
# Imports
# -------------------------------
import json
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import xgboost as xgb
import numpy as np

# -------------------------------
# Spark Session (Read-heavy, safe)
# -------------------------------
spark = (
    SparkSession.builder
    .appName("Backblaze-Operational-Impact")
    .config("spark.driver.memory", "8g")
    .config("spark.executor.memory", "8g")
    .config("spark.sql.shuffle.partitions", "200")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# -------------------------------
# Load Test Dataset (FULL SCALE)
# -------------------------------
# This represents "future unseen data"
test_df = spark.read.parquet(
    "data/model_ready/test.parquet"
)

print("Test rows:", test_df.count())
print("Test positive rows:", test_df.filter(F.col("failure_next_24h") == 1).count())

# -------------------------------
# Load Trained XGBoost Model
# -------------------------------
model = xgb.Booster()
model.load_model("models/xgboost_backblaze.json")

# -------------------------------
# Load Threshold Metadata
# -------------------------------
with open("models/threshold.json") as f:
    threshold_meta = json.load(f)

THRESHOLD = threshold_meta["threshold"]
TARGET_RECALL = threshold_meta["target_recall"]

print(f"Using threshold: {THRESHOLD:.6f}")
print(f"Target recall: {TARGET_RECALL:.2f}")

# -------------------------------
# Feature Columns (MUST MATCH TRAINING)
# -------------------------------
FEATURES = [
    "smart_5_raw",
    "smart_187_raw",
    "smart_188_raw",
    "smart_197_raw",
    "smart_198_raw",
    "smart_194_raw"
]

TARGET = "failure_next_24h"

# -------------------------------
# Spark → XGBoost Inference (UDF)
# -------------------------------
# We score in Spark, but inference happens via XGBoost
# This mimics how production batch scoring works

def predict_proba_udf(*cols):
    """
    Convert Spark row → XGBoost probability
    """
    X = np.array(cols, dtype=float).reshape(1, -1)
    dmatrix = xgb.DMatrix(X)
    return float(model.predict(dmatrix)[0])

predict_udf = F.udf(predict_proba_udf)

scored_df = test_df.withColumn(
    "failure_probability",
    predict_udf(*FEATURES)
)

# -------------------------------
# Apply Decision Threshold
# -------------------------------
scored_df = scored_df.withColumn(
    "alert",
    F.when(F.col("failure_probability") >= THRESHOLD, 1).otherwise(0)
)

# -------------------------------
# Core Operational Metrics
# -------------------------------

# Total future failures
total_failures = scored_df.filter(F.col(TARGET) == 1).count()

# Detected failures (True Positives)
true_positives = scored_df.filter(
    (F.col("alert") == 1) & (F.col(TARGET) == 1)
).count()

# False alarms
false_positives = scored_df.filter(
    (F.col("alert") == 1) & (F.col(TARGET) == 0)
).count()

# Missed failures
false_negatives = scored_df.filter(
    (F.col("alert") == 0) & (F.col(TARGET) == 1)
).count()

# -------------------------------
# Derived Metrics
# -------------------------------
recall = true_positives / total_failures if total_failures > 0 else 0.0
precision = (
    true_positives / (true_positives + false_positives)
    if (true_positives + false_positives) > 0
    else 0.0
)

# Alerts per day (operationally critical)
alerts_per_day = (
    scored_df
    .groupBy("date")
    .agg(F.sum("alert").alias("alerts"))
    .agg(F.avg("alerts"))
    .collect()[0][0]
)

# -------------------------------
# Print Operational Summary
# -------------------------------
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

# -------------------------------
# Save Alerted Drives (for Ops)
# -------------------------------
alerts_df = scored_df.filter(F.col("alert") == 1)

alerts_df.write.mode("overwrite").parquet(
    "outputs/daily_alerts.parquet"
)

print("Alert dataset written to outputs/daily_alerts.parquet")

spark.stop()
