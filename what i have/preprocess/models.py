import logging
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
# Spark MLlib imports will be added in pipeline.py

def train_logistic_regression(X_train, y_train):
    logging.info("Training Logistic Regression")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def train_decision_tree(X_train, y_train):
    logging.info("Training Decision Tree")
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    logging.info("Training Random Forest")
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    logging.info("Training XGBoost")
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    return model
