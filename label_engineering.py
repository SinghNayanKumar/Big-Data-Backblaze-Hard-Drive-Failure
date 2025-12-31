"""
Label Engineering for Predictive Maintenance
Backblaze Hard Drive Failure Dataset

Objective:
- Create a forward-looking binary label `failure_next_24h`
- Label = 1 if the drive fails the next day
- Ensure NO temporal leakage
- Be memory-safe on a single-node (WSL / laptop) Spark setup
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


# ------------------------------------------------------------------------------
# 1. Spark Session Configuration (Single-Node Safe)
# ------------------------------------------------------------------------------
# Key ideas:
# - Explicit memory limits (Spark defaults are unsafe)
# - Lower shuffle width to avoid executor explosion
# - Disable Arrow (not helpful for window-heavy workloads)
spark = (
    SparkSession.builder
    .appName("BackblazeLabelEngineering")
    .config("spark.driver.memory", "8g")
    .config("spark.executor.memory", "8g")
    .config("spark.sql.shuffle.partitions", "64")
    .config("spark.sql.execution.arrow.pyspark.enabled", "false")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ------------------------------------------------------------------------------
# 2. Load Preprocessed Dataset
# ------------------------------------------------------------------------------
# This parquet should already:
# - Contain one row per drive per day
# - Be cleaned and schema-consistent
df = spark.read.parquet(
    "data/processed/backblaze_q3_2025.parquet"
)

print("Initial row count:", df.count())


# ------------------------------------------------------------------------------
# 3. Minimal Column Selection (Critical for Memory)
# ------------------------------------------------------------------------------
# We aggressively prune columns BEFORE any shuffle or window operation.
# This is the single most important memory optimization.
base_cols = [
    "date",
    "serial_number",
    "model",
    "failure"
]

smart_cols = [
    "smart_5_raw",    # Reallocated Sectors
    "smart_187_raw",  # Reported Uncorrectable Errors
    "smart_188_raw",  # Command Timeout
    "smart_197_raw",  # Current Pending Sector Count
    "smart_198_raw",  # Offline Uncorrectable
    "smart_194_raw"   # Temperature (°C)
]

df = df.select(base_cols + smart_cols)


# ------------------------------------------------------------------------------
# 4. Explicit Type Casting (Avoid Silent Spark Issues)
# ------------------------------------------------------------------------------
df = (
    df
    .withColumn("date", F.col("date").cast("date"))
    .withColumn("failure", F.col("failure").cast("int"))
)

for c in smart_cols:
    df = df.withColumn(c, F.col(c).cast("double"))


# ------------------------------------------------------------------------------
# 5. Repartition BEFORE Windowing (OOM Fix #1)
# ------------------------------------------------------------------------------
# Why:
# - Window functions require shuffle + sort
# - Partitioning by serial_number ensures drive-local computation
# - Limits how much data each executor must hold in memory
df = df.repartition(64, "serial_number")

# Materialize the repartition to prevent lineage explosion
df = df.persist()
df.count()


# ------------------------------------------------------------------------------
# 6. Temporal Window Definition (Per Drive)
# ------------------------------------------------------------------------------
drive_window = (
    Window
    .partitionBy("serial_number")
    .orderBy("date")
)


# ------------------------------------------------------------------------------
# 7. Forward-Looking Label Creation
# ------------------------------------------------------------------------------
# If a drive fails on day T+1 → label day T as 1
df = df.withColumn(
    "failure_next_24h",
    F.when(
        F.lead("failure", 1).over(drive_window) == 1,
        1
    ).otherwise(0)
)


# ------------------------------------------------------------------------------
# 8. Leakage Prevention
# ------------------------------------------------------------------------------
# Remove rows where failure == 1 (day-of-failure rows)
# These contain post-failure information and are not actionable
df = df.filter(F.col("failure") == 0)


# ------------------------------------------------------------------------------
# 9. Sanity Checks (Mandatory in Industry)
# ------------------------------------------------------------------------------
# Validate class imbalance and label distribution
df.groupBy("failure_next_24h").count().show()


# Spot-check one failing drive timeline
example_serial = (
    df.filter("failure_next_24h = 1")
      .select("serial_number")
      .limit(1)
      .collect()[0][0]
)

df.filter(F.col("serial_number") == example_serial) \
  .orderBy("date") \
  .select("date", "failure", "failure_next_24h") \
  .show(10, truncate=False)


# ------------------------------------------------------------------------------
# 10. Parquet Write Tuning (OOM Fix #2)
# ------------------------------------------------------------------------------
# We tune Parquet writer settings to:
# - Reduce heap pressure
# - Avoid large row-group buffering
spark.conf.set("spark.sql.parquet.compression.codec", "snappy")
spark.conf.set("spark.sql.parquet.block.size", 64 * 1024 * 1024)  # 64 MB
spark.conf.set("spark.sql.parquet.page.size", 8 * 1024 * 1024)   # 8 MB

# Reduce number of concurrent writers
df = df.coalesce(32)


# ------------------------------------------------------------------------------
# 11. Persist Labeled Dataset
# ------------------------------------------------------------------------------
df.write.mode("overwrite").parquet(
    "data/processed/backblaze_q3_2025_labeled.parquet"
)

print("Labeled Parquet written successfully (OOM-safe)")
