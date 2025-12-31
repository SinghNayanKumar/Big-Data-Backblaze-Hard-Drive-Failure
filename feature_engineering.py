"""
Feature Engineering for Predictive Maintenance
Backblaze Hard Drive Failure Dataset

Input:
- Labeled dataset with `failure_next_24h`
- One row per drive per day

Output:
- Feature-engineered dataset with rolling statistics
- Safe for ML training (no leakage)

Design Principles:
- STRICTLY backward-looking features
- Per-drive temporal consistency
- Memory-safe Spark execution
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


# ------------------------------------------------------------------------------
# 1. Spark Session Configuration
# ------------------------------------------------------------------------------
# Conservative settings for single-node execution
spark = (
    SparkSession.builder
    .appName("BackblazeFeatureEngineering")
    .config("spark.driver.memory", "8g")
    .config("spark.executor.memory", "8g")
    .config("spark.sql.shuffle.partitions", "64")
    .config("spark.sql.execution.arrow.pyspark.enabled", "false")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ------------------------------------------------------------------------------
# 2. Load Labeled Dataset
# ------------------------------------------------------------------------------
df = spark.read.parquet(
    "data/processed/backblaze_q3_2025_labeled.parquet"
)

print("Input rows:", df.count())


# ------------------------------------------------------------------------------
# 3. Feature Column Selection
# ------------------------------------------------------------------------------
# Keep only features we will actually transform
smart_cols = [
    "smart_5_raw",    # Reallocated sectors
    "smart_187_raw",  # Uncorrectable errors
    "smart_188_raw",  # Command timeout
    "smart_197_raw",  # Pending sectors
    "smart_198_raw",  # Offline uncorrectable
    "smart_194_raw"   # Temperature
]

base_cols = [
    "date",
    "serial_number",
    "model",
    "failure_next_24h"
]

df = df.select(base_cols + smart_cols)


# ------------------------------------------------------------------------------
# 4. Repartition Before Heavy Window Ops (OOM Fix)
# ------------------------------------------------------------------------------
# Ensure all rows for a drive stay together
df = df.repartition(64, "serial_number")

# Materialize
df = df.persist()
df.count()


# ------------------------------------------------------------------------------
# 5. Window Definitions (Backward-Looking Only)
# ------------------------------------------------------------------------------
# All windows END at the current row (no future access)

drive_window = (
    Window
    .partitionBy("serial_number")
    .orderBy("date")
)

# 7-day rolling window
window_7d = drive_window.rowsBetween(-6, 0)

# 14-day rolling window
window_14d = drive_window.rowsBetween(-13, 0)


# ------------------------------------------------------------------------------
# 6. Rolling Statistical Features
# ------------------------------------------------------------------------------
# These capture trends, volatility, and degradation patterns

for c in smart_cols:
    df = (
        df
        # Rolling means
        .withColumn(f"{c}_mean_7d", F.avg(c).over(window_7d))
        .withColumn(f"{c}_mean_14d", F.avg(c).over(window_14d))

        # Rolling standard deviation (instability indicator)
        .withColumn(f"{c}_std_7d", F.stddev(c).over(window_7d))

        # Rolling max (spike detection)
        .withColumn(f"{c}_max_7d", F.max(c).over(window_7d))
    )


# ------------------------------------------------------------------------------
# 7. Trend / Delta Features
# ------------------------------------------------------------------------------
# Measures rate of change (early failure signal)

for c in smart_cols:
    df = df.withColumn(
        f"{c}_delta_1d",
        F.col(c) - F.lag(c, 1).over(drive_window)
    )


# ------------------------------------------------------------------------------
# 8. Age-Based Features
# ------------------------------------------------------------------------------
# How long the drive has been active in the dataset

df = df.withColumn(
    "drive_age_days",
    F.row_number().over(drive_window)
)


# ------------------------------------------------------------------------------
# 9. Null Handling (Early Rows)
# ------------------------------------------------------------------------------
# Rolling windows produce nulls for first few days
# Strategy:
# - Forward-fill is NOT allowed (would leak)
# - We fill with 0 (model learns "early-life" behavior)

df = df.fillna(0)


# ------------------------------------------------------------------------------
# 10. Sanity Checks
# ------------------------------------------------------------------------------
print("Final row count:", df.count())

df.select(
    "failure_next_24h",
    "drive_age_days"
).groupBy("failure_next_24h").count().show()


# ------------------------------------------------------------------------------
# 11. Write Feature-Engineered Dataset (OOM Safe)
# ------------------------------------------------------------------------------
spark.conf.set("spark.sql.parquet.compression.codec", "snappy")
spark.conf.set("spark.sql.parquet.block.size", 64 * 1024 * 1024)
spark.conf.set("spark.sql.parquet.page.size", 8 * 1024 * 1024)

df = df.coalesce(32)

df.write.mode("overwrite").parquet(
    "data/processed/backblaze_q3_2025_features.parquet"
)

print("Feature-engineered dataset written successfully")
