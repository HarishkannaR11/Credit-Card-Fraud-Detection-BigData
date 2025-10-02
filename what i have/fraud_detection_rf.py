import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
df = pd.read_csv('creditcard.csv')  # make sure this file exists
X = df.drop('Class', axis=1)
y = df['Class']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train RandomForestClassifier
clf = RandomForestClassifier(random_state=42, class_weight='balanced')  # handles imbalance
clf.fit(X_train, y_train)

# Feature importance
feat_importances = pd.Series(clf.feature_importances_, index=X.columns)
feat_importances = feat_importances.sort_values(ascending=False)
print("Top features:\n", feat_importances.head(10))

# Predict probabilities
y_scores = clf.predict_proba(X_test)[:,1]  # probability of class 1 (fraud)

# Compute precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

# Plot precision-recall curve
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

# Choose a threshold (example: max recall where precision >= 0.8)
desired_precision = 0.8
threshold_index = np.argmax(precision >= desired_precision)
optimal_threshold = thresholds[threshold_index]
print("Optimal threshold for precision >= 0.8:", optimal_threshold)

# Apply threshold
y_pred_custom = (y_scores >= optimal_threshold).astype(int)

# Evaluate
print(confusion_matrix(y_test, y_pred_custom))
print(classification_report(y_test, y_pred_custom))
