# 02_baseline.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, roc_auc_score, average_precision_score
from sklearn.utils import class_weight
import joblib

# load processed
train = pd.read_csv("train_processed.csv")
test  = pd.read_csv("test_processed.csv")
X_train = train.drop(columns=['Class'])
y_train = train['Class']
X_test  = test.drop(columns=['Class'])
y_test  = test['Class']

# Option A: Logistic Regression with class weights
clf_lr = LogisticRegression(max_iter=1000, class_weight='balanced', solver='saga')
clf_lr.fit(X_train, y_train)
pred_lr = clf_lr.predict(X_test)
proba_lr = clf_lr.predict_proba(X_test)[:,1]

print("LogReg report:")
print(classification_report(y_test, pred_lr))
print("ROC-AUC:", roc_auc_score(y_test, proba_lr))
print("PR-AUC:", average_precision_score(y_test, proba_lr))

# Option B: Random Forest with balanced class weight
clf_rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', n_jobs=-1, random_state=42)
clf_rf.fit(X_train, y_train)
pred_rf = clf_rf.predict(X_test)
proba_rf = clf_rf.predict_proba(X_test)[:,1]

print("RandomForest report:")
print(classification_report(y_test, pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, proba_rf))
print("PR-AUC:", average_precision_score(y_test, proba_rf))

# Save your best model
joblib.dump(clf_rf, "rf_model.joblib")
print("Saved rf_model.joblib")
