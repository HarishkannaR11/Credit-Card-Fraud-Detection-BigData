import logging
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

def evaluate_model(model, X_test, y_test):
    logging.info("Evaluating model")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    metrics = {
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else None
    }
    return metrics
