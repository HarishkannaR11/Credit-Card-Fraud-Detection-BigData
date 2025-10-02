import logging
from pyspark.sql import SparkSession

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def get_spark_session(app_name='FraudDetection'):
    return SparkSession.builder.appName(app_name).getOrCreate()

def load_from_hdfs(spark, hdfs_path):
    logging.info(f"Loading data from HDFS: {hdfs_path}")
    return spark.read.csv(hdfs_path, header=True, inferSchema=True)
