import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
df = pd.read_csv('survey lung cancer.csv')

# Preprocess data
# Encode 'GENDER' column: M=1, F=0
df['GENDER'] = df['GENDER'].map({'M': 1, 'F': 0})

# Encode target variable: YES=1, NO=0
df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'YES': 1, 'NO': 0})

# Features and target
X = df.drop('LUNG_CANCER', axis=1)
y = df['LUNG_CANCER']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save model
with open('l_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved as l_model.pkl")
