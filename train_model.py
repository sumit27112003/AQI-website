import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Create models directory if not exists
if not os.path.exists('models'):
    os.makedirs('models')

print("Loading Data...")
# Ensure your CSV is in the 'data' folder as per structure
df = pd.read_csv("data/station_hour.csv", low_memory=False)

# --- Feature Engineering (From your notebook) ---
print("Preprocessing...")
df['Datetime'] = pd.to_datetime(df['Datetime'])
df['Year'] = df['Datetime'].dt.year
df['Month'] = df['Datetime'].dt.month
df['Day'] = df['Datetime'].dt.day
df['Hour'] = df['Datetime'].dt.hour
df = df.drop(columns="Datetime")

# Dropping columns with high nulls or IDs
# Note: Dropping Xylene as per your notebook analysis (80% nulls)
df = df.drop(columns=["Xylene", "StationId"])

# Imputing Nulls with Median (From your notebook)
cols = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "AQI"]
for col in cols:
    if col in df.columns:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

# Drop rows where Target (AQI_Bucket) is still null to ensure clean training
df = df.dropna(subset=['AQI_Bucket'])

# Encoding Target
le = LabelEncoder()
df['AQI_Bucket'] = le.fit_transform(df['AQI_Bucket'].astype(str))

# Define Features and Target
# Dropping AQI from features as it is directly correlated to the bucket (target leakage)
X = df.drop(columns=["AQI_Bucket", "AQI"]) 
y = df["AQI_Bucket"]

# --- Train Model ---
print("Training Model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)
model = DecisionTreeClassifier(max_depth=10, random_state=42)
model.fit(X_train, y_train)

# --- Save Artifacts ---
print("Saving artifacts...")
joblib.dump(model, 'models/aqi_decision_tree.pkl')
joblib.dump(le, 'models/label_encoder.pkl')

# Save median values to handle missing inputs on the website
medians = X.median().to_dict()
joblib.dump(medians, 'models/medians.pkl')

print("Done! Model saved in 'models/' folder.")
