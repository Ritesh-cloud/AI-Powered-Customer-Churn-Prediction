from flask import Flask, render_template, request
import pickle
import pandas as pd
from src.preprocess import preprocess_single_input

app = Flask(__name__)

# -------------------------------
# Load trained model, encoders, and column order
# -------------------------------
model = pickle.load(open("new project/churn_model.pkl", "rb"))
encoders = pickle.load(open("new project/encoders.pkl", "rb"))
columns = pickle.load(open("new project/columns.pkl", "rb"))  # exact training feature order

# -------------------------------
# Home page
# -------------------------------
@app.route('/')
def home():
    return render_template("index.html")

# -------------------------------
# Prediction route
# -------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ---------------------------
        # 1️⃣ Collect all 19 features from the form
        # ---------------------------
        form_data = {
            'gender': request.form['gender'],
            'SeniorCitizen': request.form['SeniorCitizen'],
            'Partner': request.form['Partner'],
            'Dependents': request.form['Dependents'],
            'PhoneService': request.form['PhoneService'],
            'MultipleLines': request.form['MultipleLines'],
            'InternetService': request.form['InternetService'],
            'OnlineSecurity': request.form['OnlineSecurity'],
            'OnlineBackup': request.form['OnlineBackup'],
            'DeviceProtection': request.form['DeviceProtection'],
            'TechSupport': request.form['TechSupport'],
            'StreamingTV': request.form['StreamingTV'],
            'StreamingMovies': request.form['StreamingMovies'],
            'Contract': request.form['Contract'],
            'PaperlessBilling': request.form['PaperlessBilling'],
            'PaymentMethod': request.form['PaymentMethod'],
            'tenure': float(request.form['tenure']),
            'MonthlyCharges': float(request.form['MonthlyCharges']),
            'TotalCharges': float(request.form['TotalCharges'])
        }

        # ---------------------------
        # 2️⃣ Preprocess input into 19-feature DataFrame
        # ---------------------------
        input_df = preprocess_single_input(form_data, encoders)

        # ---------------------------
        # 3️⃣ Reorder columns exactly as training
        # ---------------------------
        input_df = input_df[columns]

        # ---------------------------
        # 4️⃣ Make prediction
        # ---------------------------
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0][1]  # probability of churn

        # ---------------------------
        # 5️⃣ Prepare result message
        # ---------------------------
        if prediction == 1:
            result = f"⚠️ Customer likely to churn! (Churn Probability: {prediction_proba*100:.2f}%)"
        else:
            result = f"✅ Customer likely to stay. (Churn Probability: {prediction_proba*100:.2f}%)"

        return render_template("result.html", prediction_text=result)

    except Exception as e:
        return f"Error: {e}"

# -------------------------------
# Run Flask app
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
