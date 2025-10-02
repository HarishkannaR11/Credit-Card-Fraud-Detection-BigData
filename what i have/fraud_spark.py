from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

spark = SparkSession.builder.appName("FraudDetection").getOrCreate()

# Read from HDFS
df = spark.read.csv("hdfs:///user/krisharish/data/creditcard_processed.csv", header=True, inferSchema=True)

# Features & label
feature_cols = [c for c in df.columns if c != "Class"]  # "Class" is fraud label
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
data = assembler.transform(df).select("features", "Class")

# Train/test split
train, test = data.randomSplit([0.8, 0.2], seed=42)

# Logistic regression model
lr = LogisticRegression(labelCol="Class", featuresCol="features")
model = lr.fit(train)

# Evaluate
preds = model.transform(test)
evaluator = BinaryClassificationEvaluator(labelCol="Class")
print("AUC:", evaluator.evaluate(preds))

