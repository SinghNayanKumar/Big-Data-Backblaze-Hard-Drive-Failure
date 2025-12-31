"""
===========================================================
Backblaze Hard Drive Failure Prediction
SHAP Explanations – Alert-Centric (Option A)
===========================================================

GOAL
----
Explain WHY the model is firing alerts:
- Which SMART features dominate alert decisions?
- Are alerts driven by physically meaningful signals?
- Do alerts look reasonable or pathological?

SCOPE
-----
- Explain ONLY rows where alert == 1
- Spark-first filtering
- Pandas only for a small, controlled sample

IMPORTANT
---------
- No retraining
- No threshold tuning
- No Spark UDF SHAP (NOT SAFE)
===========================================================
"""

# -------------------------------
# Imports
# -------------------------------
import xgboost as xgb
import shap
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# -------------------------------
# Spark Session
# -------------------------------
spark = (
    SparkSession.builder
    .appName("Backblaze-SHAP-Alerts")
    .config("spark.driver.memory", "8g")
    .config("spark.executor.memory", "8g")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# -------------------------------
# Constants
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
THRESHOLD = 0.136154   # use the same deployment threshold
MAX_SHAP_ROWS = 5000   # HARD CAP (memory-safe)

# -------------------------------
# Load Test Data
# -------------------------------
test_df = spark.read.parquet("data/model_ready/test.parquet")

# -------------------------------
# Load Trained Model
# -------------------------------
model = xgb.Booster()
model.load_model("models/xgboost_backblaze.json")

# -------------------------------
# Spark → XGBoost Inference UDF
# -------------------------------
def predict_proba_udf(*cols):
    X = np.array(cols, dtype=float).reshape(1, -1)
    dmatrix = xgb.DMatrix(X, feature_names=FEATURES)
    return float(model.predict(dmatrix)[0])

predict_udf = F.udf(predict_proba_udf)

scored_df = test_df.withColumn(
    "failure_probability",
    predict_udf(*FEATURES)
)

scored_df = scored_df.withColumn(
    "alert",
    F.when(F.col("failure_probability") >= THRESHOLD, 1).otherwise(0)
)

# -------------------------------
# Filter Alerts Only (Spark-side)
# -------------------------------
alerts_df = scored_df.filter(F.col("alert") == 1)

total_alerts = alerts_df.count()
print(f"Total alerts in test set: {total_alerts}")

# -------------------------------
# Sample Alerts for SHAP
# -------------------------------
sample_fraction = min(1.0, MAX_SHAP_ROWS / max(total_alerts, 1))

alerts_sample_df = alerts_df.sample(
    fraction=sample_fraction,
    seed=42
)

alerts_sample_df = alerts_sample_df.limit(MAX_SHAP_ROWS)

print(f"SHAP sample size: {alerts_sample_df.count()}")

# -------------------------------
# Convert to Pandas (SAFE)
# -------------------------------
alerts_pd = alerts_sample_df.select(
    FEATURES + [TARGET, "failure_probability"]
).toPandas()

X_shap = alerts_pd[FEATURES]

# -------------------------------
# SHAP Explainer
# -------------------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_shap)

# -------------------------------
# Global Explanation
# -------------------------------
print("\nGenerating SHAP summary plot...")
shap.summary_plot(
    shap_values,
    X_shap,
    show=False
)

# Save plot
import matplotlib.pyplot as plt
plt.tight_layout()
plt.savefig("outputs/shap_alert_summary.png", dpi=150)
plt.close()

print("SHAP summary saved to outputs/shap_alert_summary.png")

# -------------------------------
# Save SHAP Values (Optional, for analysis)
# -------------------------------
shap_df = pd.DataFrame(
    shap_values,
    columns=[f"shap_{f}" for f in FEATURES]
)

shap_df["failure_probability"] = alerts_pd["failure_probability"].values
shap_df[TARGET] = alerts_pd[TARGET].values

shap_df.to_csv(
    "outputs/shap_alert_values.csv",
    index=False
)

print("SHAP values saved to outputs/shap_alert_values.parquet")

spark.stop()
