import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ----------------------------------------------------------------------
# 🔹 TRAINING DATA PREPROCESSING FUNCTION
# ----------------------------------------------------------------------
def load_and_preprocess(filepath = "C:/Users/rites/Downloads/archive (2)/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    """
    Load and preprocess the Telco Customer Churn dataset for training.
    Returns the cleaned DataFrame and fitted encoders.
    """
    df = pd.read_csv(filepath)

    # Replace spaces with NaN, handle missing values
    df['TotalCharges'] = df['TotalCharges'].replace(" ", None)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna()

    # Encode target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Columns to encode
    cat_cols = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]

    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # Drop customerID
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)

    return df, encoders


# ----------------------------------------------------------------------
# 🔹 SINGLE ROW (USER INPUT) PREPROCESSING FUNCTION
# ----------------------------------------------------------------------
def preprocess_single_input(form_data, encoders):
    """
    Preprocess a single user input from Flask form using the same encoders.
    Returns a DataFrame with the same columns as used in training.
    """

    # Convert form data to DataFrame
    df = pd.DataFrame([form_data])

    # Ensure numeric fields are correct
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Apply the same encoders used during training
    for col, le in encoders.items():
        if col in df.columns:
            try:
                df[col] = le.transform(df[col])
            except ValueError:
                # Handle unseen category gracefully
                df[col] = 0

    # Make sure all expected columns are present
    expected_cols = list(encoders.keys()) + ['tenure', 'MonthlyCharges', 'TotalCharges']
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns
    df = df[expected_cols]

    return df
