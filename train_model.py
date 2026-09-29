import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# Paths
DATA_PATH = os.path.join("data", "asthma_disease_data.csv")
MODEL_PATH = os.path.join("models", "model.pkl")

os.makedirs("models", exist_ok=True)

# Load Dataset
df = pd.read_csv(DATA_PATH)

# Select Important Features
df = df[[
    "Age",
    "Gender",
    "BMI",
    "Smoking",
    "HistoryOfAllergies",
    "FamilyHistoryAsthma",
    "Wheezing",
    "ShortnessOfBreath",
    "ChestTightness",
    "Coughing",
    "ExerciseInduced",
    "Diagnosis"
]]

# Features & Target
X = df.drop("Diagnosis", axis=1)
y = df["Diagnosis"]

# Split before SMOTE (Best Practice)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Apply SMOTE only on training data
smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("Before SMOTE")
print(y_train.value_counts())

print("\nAfter SMOTE")
print(y_train_smote.value_counts())

# Train Model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train_smote, y_train_smote)

# Prediction
y_pred = model.predict(X_test)

print("\nAccuracy :", accuracy_score(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# Feature Importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

print("\nFeature Importance")
print(importance.sort_values(by="Importance", ascending=False))

# Save Model
joblib.dump(model, MODEL_PATH)

print("\n✅ Model Trained Successfully")