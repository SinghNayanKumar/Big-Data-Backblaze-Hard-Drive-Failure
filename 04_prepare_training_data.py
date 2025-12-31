from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# -------------------------------
# Spark Session (Memory Safe)
# -------------------------------
spark = (
    SparkSession.builder
    .appName("Backblaze-Imbalance-Handling")
    .config("spark.driver.memory", "8g")
    .config("spark.executor.memory", "8g")
    .config("spark.sql.shuffle.partitions", "200")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# -------------------------------
# Load Labeled Dataset
# -------------------------------
df = spark.read.parquet(
    "data/processed/backblaze_q3_2025_labeled.parquet"
)

print("Total rows:", df.count())

# -------------------------------
# Time-Based Split to prevent future leakage(CRITICAL)
# -------------------------------
# July to aug25 → Train
# August25 to sep15 → Validation
# September15 onwards → Test

train_df = df.filter("date < '2025-08-25'")
val_df   = df.filter("date >= '2025-08-25' AND date < '2025-09-15'")
test_df  = df.filter("date >= '2025-09-15'")

print("Train rows:", train_df.count())
print("Val rows:", val_df.count())
print("Test rows:", test_df.count())

# -------------------------------
# Handle Extreme Imbalance
# -------------------------------
# Separate positives & negatives
train_pos = train_df.filter("failure_next_24h = 1")
train_neg = train_df.filter("failure_next_24h = 0")

pos_count = train_pos.count()
neg_count = train_neg.count()

print(f"Positive samples: {pos_count}")
print(f"Negative samples: {neg_count}")
# pos:neg = 695 : 17,761,356 ≈ 1 : 25,000  
# target ratio for tree based models is 1 : 50 to 1 : 200
# target ~70k–140k negatives for 695 positives



# Downsample negatives (100:1 ratio)
NEGATIVE_RATIO = 100

train_neg_sampled = train_neg.sample(
    fraction=min(1.0, (pos_count * NEGATIVE_RATIO) / neg_count),
    seed=42
)

train_balanced = train_pos.unionByName(train_neg_sampled)

print("Balanced train rows:", train_balanced.count())

# -------------------------------
# Persist Model-Ready Data
# -------------------------------
train_balanced.write.mode("overwrite").parquet(
    "data/model_ready/train.parquet"
)

val_df.write.mode("overwrite").parquet(
    "data/model_ready/val.parquet"
)

test_df.write.mode("overwrite").parquet(
    "data/model_ready/test.parquet"
)

print("Model-ready datasets written successfully")

spark.stop()
