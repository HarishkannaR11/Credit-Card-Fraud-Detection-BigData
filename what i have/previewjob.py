import joblib

# Load and display the content
data = joblib.load('rf_model.joblib')
print("Content of the joblib file:")
print(data)
