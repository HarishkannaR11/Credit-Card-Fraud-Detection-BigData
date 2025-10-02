from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# --------------------------
# 1. Spark session
# --------------------------
spark = SparkSession.builder \
    .appName("CreditCardFraudDetection") \
    .getOrCreate()

# --------------------------
# 2. Read CSVs from HDFS
# --------------------------
train_df = spark.read.csv(
    "hdfs:///user/krisharish/creditcard_data/train_processed.csv",
    header=True, inferSchema=True
)

test_df = spark.read.csv(
    "hdfs:///user/krisharish/creditcard_data/test_processed.csv",
    header=True, inferSchema=True
)

# Quick look
train_df.show(5)
train_df.printSchema()

# --------------------------
# 3. Prepare features
# --------------------------
feature_cols = [c for c in train_df.columns if c != "Class"]  # Assuming "Class" is target
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

train_df = assembler.transform(train_df)
test_df = assembler.transform(test_df)

# --------------------------
# 4. Train Random Forest
# --------------------------
rf = RandomForestClassifier(featuresCol="features", labelCol="Class", numTrees=100)
rf_model = rf.fit(train_df)

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
# 7. Save trained model
# --------------------------
rf_model.save("hdfs:///user/krisharish/creditcard_data/rf_model_spark")

# --------------------------
# 8. Stop Spark session
# --------------------------
spark.stop()
