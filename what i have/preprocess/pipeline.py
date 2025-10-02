import argparse
import logging
import joblib
from data_preprocessing import load_data, handle_imbalance, split_data
from feature_engineering import scale_features, apply_pca
from models import train_logistic_regression, train_decision_tree, train_random_forest, train_xgboost
from evaluation import evaluate_model
from utils import setup_logging, get_spark_session, load_from_hdfs

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description='Credit Card Fraud Detection Pipeline')
    parser.add_argument('--data', type=str, help='Path to CSV or HDFS')
    parser.add_argument('--model', type=str, choices=['logreg', 'dtree', 'rf', 'xgb', 'spark'], default='logreg')
    parser.add_argument('--imbalance', type=str, choices=['smote', 'undersample', 'none'], default='smote')
    parser.add_argument('--pca', type=int, default=None)
    parser.add_argument('--save_model', type=str, default='model.pkl')
    parser.add_argument('--save_pred', type=str, default='predictions.csv')
    args = parser.parse_args()

    if args.data.startswith('hdfs://'):
        spark = get_spark_session()
        df = load_from_hdfs(spark, args.data)
        # Spark MLlib pipeline would go here
        logging.info('Spark MLlib pipeline not yet implemented')
        return
    else:
        df = load_data(args.data)

    X = df.drop('Class', axis=1)
    y = df['Class']
    X, y = handle_imbalance(X, y, method=args.imbalance)
    X = scale_features(X)
    if args.pca:
        X = apply_pca(X, n_components=args.pca)
    X_train, X_test, y_train, y_test = split_data(X, y)

    if args.model == 'logreg':
        model = train_logistic_regression(X_train, y_train)
    elif args.model == 'dtree':
        model = train_decision_tree(X_train, y_train)
    elif args.model == 'rf':
        model = train_random_forest(X_train, y_train)
    elif args.model == 'xgb':
        model = train_xgboost(X_train, y_train)
    else:
        logging.info('Spark MLlib pipeline not yet implemented')
        return

    metrics = evaluate_model(model, X_test, y_test)
    logging.info(f"Evaluation metrics: {metrics}")
    joblib.dump(model, args.save_model)
    logging.info(f"Model saved to {args.save_model}")
    import pandas as pd
    pd.DataFrame({'y_true': y_test, 'y_pred': model.predict(X_test)}).to_csv(args.save_pred, index=False)
    logging.info(f"Predictions saved to {args.save_pred}")

if __name__ == '__main__':
    main()
