from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the ML model
model = pickle.load(open('loan_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')  # Main HTML template for the app

@app.route('/predict', methods=['POST'])
def predict():
    # Collect form data
    features = [int(request.form[key]) for key in ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'CreditHistory', 'LoanTerm', 'PropertyArea']]
    prediction = model.predict([features])[0]
    prediction_text = "Approved" if prediction == 1 else "Rejected"
    return render_template('result.html', prediction=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
