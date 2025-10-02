from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# --------------------------
# 1. Start Spark session
# --------------------------
spark = SparkSession.builder \
    .appName("LoadCreditCardRFModel") \
    .getOrCreate()

# --------------------------
# 2. Load saved Random Forest model from HDFS
# --------------------------
rf_model = RandomForestClassificationModel.load(
    "hdfs:///user/krisharish/creditcard_data/rf_model_spark"
)

# --------------------------
# 3. Read test data from HDFS
# --------------------------
test_df = spark.read.csv(
    "hdfs:///user/krisharish/creditcard_data/test_processed.csv",
    header=True, inferSchema=True
)

# --------------------------
# 4. Prepare features
# --------------------------
from pyspark.ml.feature import VectorAssembler

feature_cols = [c for c in test_df.columns if c != "Class"]  # Exclude target
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
test_df = assembler.transform(test_df)

# --------------------------
# 5. Make predictions
# --------------------------
predictions = rf_model.transform(test_df)
predictions.select("Class", "prediction", "probability").show(5)

# --------------------------
# 6. Evaluate model
# --------------------------
evaluator = BinaryClassificationEvaluator(labelCol="Class", metricName="areaUnderROC")
roc_auc = evaluator.evaluate(predictions)
print(f"Test ROC AUC: {roc_auc}")

# --------------------------
# 7. Inspect model
# --------------------------
# Number of trees
print(f"Number of trees: {len(rf_model.trees)}")

# Feature importances
print(f"Feature importances: {rf_model.featureImportances}")

# --------------------------
# 8. Stop Spark session
# --------------------------
spark.stop()
