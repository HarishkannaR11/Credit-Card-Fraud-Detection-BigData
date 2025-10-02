from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import sqlite3
import time
import os
from threading import Lock

app = Flask(__name__)

# Load your trained model
MODEL_PATH = 'rf_model.joblib'  # Update with your actual model path
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Thread-safe metrics storage
metrics_lock = Lock()
metrics = {
    'total_predictions': 0,
    'fraud_detected': 0,
    'legitimate_detected': 0,
    'fraud_rate': 0.0,
    'last_prediction_time': None,
    'average_confidence': 0.0,
    'predictions_per_hour': 0,
    'model_status': 'healthy' if model else 'error',
    'uptime_start': datetime.now().isoformat()
}

# Database setup for detailed metrics
DB_PATH = 'model_metrics.db'

def init_database():
    """Initialize SQLite database for storing metrics"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            prediction INTEGER,
            fraud_probability REAL,
            legitimate_probability REAL,
            confidence REAL,
            processing_time_ms REAL
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS hourly_stats (
            hour_timestamp DATETIME PRIMARY KEY,
            total_predictions INTEGER,
            fraud_predictions INTEGER,
            avg_fraud_probability REAL,
            avg_confidence REAL,
            avg_processing_time_ms REAL
        )
    ''')
    
    conn.commit()
    conn.close()

def log_prediction(prediction, probabilities, processing_time):
    """Log individual prediction to database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        fraud_prob = float(probabilities[1]) if len(probabilities) > 1 else 0.0
        legit_prob = float(probabilities[0]) if len(probabilities) > 0 else 0.0
        confidence = max(fraud_prob, legit_prob)
        
        cursor.execute('''
            INSERT INTO predictions 
            (prediction, fraud_probability, legitimate_probability, confidence, processing_time_ms)
            VALUES (?, ?, ?, ?, ?)
        ''', (int(prediction), fraud_prob, legit_prob, confidence, processing_time * 1000))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error logging prediction: {e}")

def update_metrics(prediction, probabilities):
    """Update in-memory metrics"""
    global metrics
    with metrics_lock:
        metrics['total_predictions'] += 1
        
        if prediction == 1:
            metrics['fraud_detected'] += 1
        else:
            metrics['legitimate_detected'] += 1
        
        metrics['fraud_rate'] = metrics['fraud_detected'] / metrics['total_predictions']
        metrics['last_prediction_time'] = datetime.now().isoformat()
        
        # Update average confidence
        confidence = max(probabilities) if len(probabilities) > 0 else 0.0
        current_avg = metrics.get('average_confidence', 0.0)
        total = metrics['total_predictions']
        metrics['average_confidence'] = ((current_avg * (total - 1)) + confidence) / total

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        start_time = time.time()
        
        # Parse input data
        data = request.json
        if 'features' not in data:
            return jsonify({'error': 'Missing features in request'}), 400
        
        features = np.array(data['features']).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        processing_time = time.time() - start_time
        
        # Update metrics and log
        update_metrics(prediction, probabilities)
        log_prediction(prediction, probabilities, processing_time)
        
        response = {
            'prediction': int(prediction),
            'prediction_label': 'fraud' if prediction == 1 else 'legitimate',
            'fraud_probability': float(probabilities[1]) if len(probabilities) > 1 else 0.0,
            'legitimate_probability': float(probabilities[0]) if len(probabilities) > 0 else 0.0,
            'confidence': float(max(probabilities)),
            'processing_time_ms': processing_time * 1000,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/metrics', methods=['GET'])
def get_metrics():
    """Metrics endpoint for Grafana"""
    with metrics_lock:
        current_metrics = metrics.copy()
    
    # Add uptime calculation
    uptime_start = datetime.fromisoformat(current_metrics['uptime_start'])
    uptime_seconds = (datetime.now() - uptime_start).total_seconds()
    current_metrics['uptime_seconds'] = uptime_seconds
    current_metrics['uptime_hours'] = uptime_seconds / 3600
    
    return jsonify(current_metrics)

@app.route('/metrics/detailed', methods=['GET'])
def get_detailed_metrics():
    """Detailed metrics from database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get hourly statistics
        cursor.execute('''
            SELECT 
                strftime('%Y-%m-%d %H:00:00', timestamp) as hour,
                COUNT(*) as total_predictions,
                SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END) as fraud_predictions,
                AVG(fraud_probability) as avg_fraud_prob,
                AVG(confidence) as avg_confidence,
                AVG(processing_time_ms) as avg_processing_time
            FROM predictions 
            WHERE timestamp >= datetime('now', '-24 hours')
            GROUP BY strftime('%Y-%m-%d %H:00:00', timestamp)
            ORDER BY hour DESC
            LIMIT 24
        ''')
        
        hourly_data = []
        for row in cursor.fetchall():
            hourly_data.append({
                'hour': row[0],
                'total_predictions': row[1],
                'fraud_predictions': row[2],
                'fraud_rate': row[2] / row[1] if row[1] > 0 else 0,
                'avg_fraud_probability': row[3],
                'avg_confidence': row[4],
                'avg_processing_time_ms': row[5]
            })
        
        conn.close()
        
        return jsonify({
            'hourly_stats': hourly_data,
            'query_timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if model else 'unhealthy',
        'model_loaded': model is not None,
        'database_accessible': os.path.exists(DB_PATH),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/test-prediction', methods=['GET'])
def test_prediction():
    """Test endpoint with sample data"""
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Generate random test data (adjust features count based on your model)
    n_features = model.n_features_in_
    test_features = np.random.randn(n_features).tolist()
    
    # Make a test prediction
    response = predict()
    
    return jsonify({
        'message': 'Test prediction completed',
        'test_features_count': n_features,
        'sample_features': test_features[:5]  # Show first 5 features
    })

if __name__ == '__main__':
    # Initialize database
    init_database()
    print("Database initialized")
    
    print(f"Starting ML API server...")
    print(f"Model loaded: {model is not None}")
    print(f"Access metrics at: http://localhost:5000/metrics")
    print(f"Health check at: http://localhost:5000/health")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
