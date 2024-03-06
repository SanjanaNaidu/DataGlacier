import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('Iris.csv')

# Drop the 'Id' column if it exists
if 'Id' in df.columns:
    df.drop(['Id'], axis=1, inplace=True)

# Define the feature matrix X and the target y
X = df.drop(columns=['Species'])
y = df['Species']

# Encode the target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train the Logistic Regression model
lr = LogisticRegression(solver = 'lbfgs',random_state=42, max_iter=1000)
lr.fit(X_train, y_train)

# Make predictions
y_pred = lr.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
predicted_species = le.inverse_transform(y_pred)

# Save the model to disk
joblib.dump(lr, 'model.pkl')
joblib.dump(le, 'label_encoder.pkl') 