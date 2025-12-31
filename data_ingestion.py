from pyspark.sql import SparkSession
import os

os.environ["PYSPARK_PYTHON"] = os.path.abspath(".venv\\Scripts\\python.exe")
os.environ["PYSPARK_DRIVER_PYTHON"] = os.environ["PYSPARK_PYTHON"]

spark = (
    SparkSession.builder
    .appName("Backblaze-Ingestion")
    .config("spark.sql.shuffle.partitions", 200)
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")
print("Spark Session Created")

df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv("data/raw/data_Q3_2025")
)

print("Row count:", df.count())

df.write.mode("overwrite").parquet(
    "data/processed/backblaze_q3_2025.parquet"
)

print("Parquet written successfully")
