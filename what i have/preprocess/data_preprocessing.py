import logging
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

def load_data(path):
    logging.info(f"Loading data from {path}")
    return pd.read_csv(path)

def handle_imbalance(X, y, method='smote'):
    logging.info(f"Handling imbalance with {method}")
    if method == 'smote':
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X, y)
        return X_res, y_res
    elif method == 'undersample':
        # Add undersampling logic here
        pass
    else:
        return X, y

def split_data(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42)
