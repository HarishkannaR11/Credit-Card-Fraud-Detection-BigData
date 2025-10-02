# 01_preprocess.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# load
df = pd.read_csv("creditcard.csv")   # adjust path if needed

# quick checks
print(df.shape)
print(df['Class'].value_counts())

# features and target
X = df.drop(columns=['Class','Time'])   # drop Time or keep depending
y = df['Class']

# scale Amount (and maybe Time)
scaler = StandardScaler()
X['Amount'] = scaler.fit_transform(X[['Amount']])

# train-test split (stratify to keep imbalance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# save processed CSVs for Spark / HDFS later if needed
X_train.assign(Class=y_train).to_csv("train_processed.csv", index=False)
X_test.assign(Class=y_test).to_csv("test_processed.csv", index=False)
print("Saved train_processed.csv and test_processed.csv")
