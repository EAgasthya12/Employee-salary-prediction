# Employee-salary-prediction

A machine learning web app built using Random Forest and Streamlit to predict whether a person earns >50K or ≤50K based on the UCI Adult dataset.

## 🔍 Features
- Predict salary class using age, education, occupation, hours/week, and experience
- RandomForestClassifier with OneHotEncoding
- Real-time input and CSV batch prediction
- Streamlit frontend

## 🗂 Folder Contents
- `model_training.ipynb` - Model training pipeline
- `best_model.pkl` - Trained pipeline
- `app.py` - Streamlit interface
- `adult.csv` - Dataset
- `model_accuracy.txt` - Saved accuracy

## 🚀 To Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py