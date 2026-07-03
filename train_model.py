import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# Create models directory if not exists
if not os.path.exists('models'):
    os.makedirs('models')

print("Loading Data...")
# Ensure your CSV is in the 'data' folder as per structure
df = pd.read_csv("data/station_hour.csv", low_memory=False)

# --- Feature Engineering ---
print("Preprocessing...")
df['Datetime'] = pd.to_datetime(df['Datetime'])
df['Year'] = df['Datetime'].dt.year
df['Month'] = df['Datetime'].dt.month
df['Day'] = df['Datetime'].dt.day
df['Hour'] = df['Datetime'].dt.hour
df = df.drop(columns="Datetime")

# NOTE: 'Year' is included as a feature below for consistency with the
# previously deployed model. Be aware a decision tree cannot extrapolate
# beyond the years seen in training - if you query the deployed app with a
# date far outside this CSV's date range, predictions may be unreliable for
# that reason alone. Consider dropping 'Year' (keep Month/Hour for
# seasonality) if you want the model to stay reliable indefinitely.

# Dropping columns with high nulls or IDs
# Note: Dropping Xylene as per earlier analysis (~80% nulls)
df = df.drop(columns=["Xylene", "StationId"])

# Drop rows where the target is missing - do this before any imputation so
# imputation statistics aren't influenced by rows we're about to discard.
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
# stratify=y keeps the class balance consistent across train/test, which
# matters here since AQI categories are unlikely to be evenly distributed
# (e.g. far fewer "Severe" hours than "Moderate" ones).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=20, stratify=y
)

# Impute missing pollutant values using medians computed from the TRAINING
# split only, then apply those same medians to the test split. Fitting
# imputation stats on the full dataset (train+test together) before
# splitting is a form of data leakage - the test set should stay unseen
# until evaluation.
impute_cols = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene"]
for col in impute_cols:
    if col in X_train.columns:
        med = X_train[col].median()
        X_train[col] = X_train[col].fillna(med)
        X_test[col] = X_test[col].fillna(med)

model = DecisionTreeClassifier(max_depth=10, random_state=42)
model.fit(X_train, y_train)

# --- Evaluate ---
print("\nEvaluating Model...")
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {acc:.4f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print("Confusion matrix (rows=actual, cols=predicted):")
print(pd.DataFrame(confusion_matrix(y_test, y_pred), index=le.classes_, columns=le.classes_))

# --- Save Artifacts ---
print("\nSaving artifacts...")
joblib.dump(model, 'models/aqi_decision_tree.pkl')
joblib.dump(le, 'models/label_encoder.pkl')

# Medians used by the web app to pre-fill/fallback pollutant fields the user
# doesn't enter directly. Computed from the training split only (see above),
# not from train+test combined, and not from held-out test rows.
full_medians = X_train.median().to_dict()
joblib.dump(full_medians, 'models/medians.pkl')

# Explicitly persist the training feature order/names as its own artifact.
# sklearn also stores this on the model itself (model.feature_names_in_)
# whenever .fit() is called with a DataFrame, but saving it separately here
# makes the training/serving contract obvious even without inspecting the
# model object, and gives app.py a fallback if that attribute is ever
# missing (e.g. a much older sklearn version).
joblib.dump(list(X_train.columns), 'models/feature_order.pkl')

# Record the scikit-learn version used to fit the model. Unpickling a model
# with a different sklearn version than it was trained with can silently
# change behavior or raise warnings/errors - check this against your
# inference environment's `pip show scikit-learn` if predictions ever look
# off after an environment change.
with open('models/sklearn_version.txt', 'w') as f:
    f.write(sklearn.__version__)

print(f"Done! Model saved in 'models/' folder. (trained with scikit-learn {sklearn.__version__})")
