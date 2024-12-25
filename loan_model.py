import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset from CSV
df = pd.read_csv('loan_data.csv')

# Features and labels
X = df[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'CreditHistory', 'LoanTerm', 'PropertyArea']]
y = df['LoanStatus']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the trained model
with open('loan_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model trained and saved successfully!")
