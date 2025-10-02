import requests
import json
import time
import random
import numpy as np

API_BASE_URL = 'http://localhost:5000'

def test_health():
    """Test health endpoint"""
    try:
        response = requests.get(f'{API_BASE_URL}/health')
        print("Health Check:")
        print(json.dumps(response.json(), indent=2))
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_metrics():
    """Test metrics endpoint"""
    try:
        response = requests.get(f'{API_BASE_URL}/metrics')
        print("\nCurrent Metrics:")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Metrics check failed: {e}")

def generate_sample_features(n_features=30):
    """Generate random sample features for testing"""
    # Generate features that might represent credit card transaction data
    features = []
    
    # Amount (normalized)
    features.append(random.normalvariate(0, 1))
    
    # Time features (V1-V28 from PCA)
    for i in range(n_features - 1):
        features.append(random.normalvariate(0, 1))
    
    return features

def test_predictions(num_tests=10):
    """Generate test predictions"""
    print(f"\nGenerating {num_tests} test predictions...")
    
    for i in range(num_tests):
        try:
            # Generate random features (adjust count based on your model)
            features = generate_sample_features(30)  # Adjust this number
            
            payload = {'features': features}
            response = requests.post(
                f'{API_BASE_URL}/predict',
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"Prediction {i+1}: {result['prediction_label']} "
                      f"(confidence: {result['confidence']:.3f})")
            else:
                print(f"Prediction {i+1} failed: {response.text}")
            
            # Small delay between requests
            time.sleep(0.1)
            
        except Exception as e:
            print(f"Error in prediction {i+1}: {e}")
    
    print(f"\nCompleted {num_tests} test predictions!")

def main():
    print("Testing ML API...")
    
    # Test health
    if not test_health():
        print("API is not healthy. Make sure api.py is running.")
        return
    
    # Test initial metrics
    test_metrics()
    
    # Generate some test predictions
    test_predictions(20)
    
    # Check updated metrics
    print("\n" + "="*50)
    test_metrics()
    
    print("\nTest completed! Check your Grafana dashboard at http://localhost:3000")

if __name__ == '__main__':
    main()
