import streamlit as st
import pandas as pd
import joblib

# Load trained pipeline model
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")

st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Sidebar inputs
st.sidebar.header("Input Employee Details")
age = st.sidebar.slider("Age", 18, 65, 30)
education = st.sidebar.selectbox("Education Level", [
    "Bachelors", "Masters", "Doctorate", "HS-grad", "Assoc-acdm", "Assoc-voc", "Some-college"
])
occupation = st.sidebar.selectbox("Job Role", [
    "Tech-support", "Craft-repair", "Other-service", "Sales",
    "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct",
    "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv",
    "Protective-serv", "Armed-Forces"
])
hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)
experience = st.sidebar.slider("Years of Experience", 0, 47, 5)

# Prepare input DataFrame
input_df = pd.DataFrame({
    'age': [age],
    'education': [education],
    'occupation': [occupation],
    'hours.per.week': [hours_per_week],  # ✅ use correct column name
    'experience': [experience]
})

st.write("### 🔎 Input Data")
st.write(input_df)

# Predict button
if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)
    st.success(f"✅ Prediction: {prediction[0]}")

# Batch Prediction
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)

    # Check required columns
    required_cols = ['age', 'education', 'occupation', 'hours.per.week', 'experience']
    if all(col in batch_data.columns for col in required_cols):
        st.write("Uploaded data preview:", batch_data.head())
        batch_preds = model.predict(batch_data)
        batch_data['PredictedClass'] = batch_preds
        st.write("✅ Predictions:")
        st.write(batch_data.head())
        csv = batch_data.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')
    else:
        st.error("❌ CSV file is missing one or more required columns: " + ', '.join(required_cols))
