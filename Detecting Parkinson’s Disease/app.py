from flask import Flask, request, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the models and scaler
model_rf = joblib.load('model_rf.pkl')
model_svc = joblib.load('model_svc.pkl')
model_gb = joblib.load('model_gb.pkl')
scaler = joblib.load('scaler.pkl')

# Define the feature names
feature_names = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 
    'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 
    'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 
    'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Handle the input data from the form
    try:
        input_data = [float(request.form[feature]) for feature in feature_names]
    except ValueError:
        return render_template('index.html', prediction_text={"error": "Invalid input. Please ensure all fields are filled correctly."})
    
    input_df = pd.DataFrame([input_data], columns=feature_names)
    
    # Scale the input data
    standard_data = scaler.transform(input_df)
    
    # Make predictions
    prediction_rf = model_rf.predict(standard_data)[0]
    prediction_svc = model_svc.predict(standard_data)[0]
    prediction_gb = model_gb.predict(standard_data)[0]
    
    # Map the predictions to human-readable text
    prediction_text = {
        "Random Forest": "The person is healthy" if prediction_rf == 0 else "The person has Parkinson's disease",
        "SVC": "The person is healthy" if prediction_svc == 0 else "The person has Parkinson's disease",
        "Gradient Boosting": "The person is healthy" if prediction_gb == 0 else "The person has Parkinson's disease"
    }
    
    return render_template('index.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
