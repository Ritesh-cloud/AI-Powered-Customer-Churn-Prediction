import pickle
from preprocess import load_and_preprocess
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load and preprocess dataset
df, encoders = load_and_preprocess("C:/Users/rites/Downloads/archive (2)/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Split features & target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save model & encoders
columns = X.columns.tolist()  # ✅ save exact training order

pickle.dump(model, open("churn_model.pkl", "wb"))
pickle.dump(encoders, open("encoders.pkl", "wb"))
pickle.dump(columns, open("columns.pkl", "wb"))  # ✅ crucial for Flask

print("✅ Model, encoders, and column order saved successfully.")
