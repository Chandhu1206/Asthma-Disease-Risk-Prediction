import streamlit as st
import pandas as pd
import joblib
import os

# Model Path
MODEL_PATH = os.path.join("models", "model.pkl")

st.set_page_config(page_title="Asthma Risk Prediction", layout="centered")

st.title("🌬 Asthma Disease Risk Prediction")

# Load Model
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    st.error("❌ Model not found. Please run train_model.py first.")
    st.stop()

# Sidebar
option = st.sidebar.radio(
    "Choose Mode",
    ["Single Prediction", "Bulk Prediction"]
)

# SINGLE PREDICTION #
if option == "Single Prediction":

    st.subheader("🧍 Patient Details")

    age = st.slider("Age", 1, 100, 25)

    gender = st.selectbox("Gender", ["Male", "Female"])
    gender = 1 if gender == "Male" else 0

    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.0)

    smoking = st.selectbox("Smoking", ["Yes", "No"])
    smoking = 1 if smoking == "Yes" else 0

    allergy = st.selectbox("History Of Allergies", ["Yes", "No"])
    allergy = 1 if allergy == "Yes" else 0

    family = st.selectbox("Family History Of Asthma", ["Yes", "No"])
    family = 1 if family == "Yes" else 0

    wheezing = st.selectbox("Wheezing", ["Yes", "No"])
    wheezing = 1 if wheezing == "Yes" else 0

    breath = st.selectbox("Shortness Of Breath", ["Yes", "No"])
    breath = 1 if breath == "Yes" else 0

    chest = st.selectbox("Chest Tightness", ["Yes", "No"])
    chest = 1 if chest == "Yes" else 0

    cough = st.selectbox("Coughing", ["Yes", "No"])
    cough = 1 if cough == "Yes" else 0

    exercise = st.selectbox("Exercise Induced Symptoms", ["Yes", "No"])
    exercise = 1 if exercise == "Yes" else 0

    if st.button("Predict Risk"):

        sample = pd.DataFrame([[
            age,
            gender,
            bmi,
            smoking,
            allergy,
            family,
            wheezing,
            breath,
            chest,
            cough,
            exercise
        ]], columns=[
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
            "ExerciseInduced"
        ])

        prediction = model.predict(sample)[0]
        probability = model.predict_proba(sample)[0]

        st.write("### Prediction Probability")
        st.write(f"🟢 Low Risk : {probability[0]*100:.2f}%")
        st.write(f"🔴 High Risk : {probability[1]*100:.2f}%")

        if prediction == 1:
            st.error("⚠ High Risk of Asthma")
        else:
            st.success("✅ Low Risk of Asthma")

# BULK PREDICTION #
else:

    st.subheader("📂 Bulk Prediction")

    file = st.file_uploader("Upload CSV File", type=["csv"])

    if file is not None:

        df = pd.read_csv(file)

        required_columns = [
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
            "ExerciseInduced"
        ]

        missing = [col for col in required_columns if col not in df.columns]

        if missing:
            st.error(f"Missing Columns: {missing}")
        else:
            predictions = model.predict(df[required_columns])

            df["Prediction"] = predictions

            st.dataframe(df)

            csv = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "📥 Download Predictions",
                csv,
                "predictions.csv",
                "text/csv"
            )