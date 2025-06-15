import pandas as pd
import joblib

# Load the saved model
model = joblib.load("model.pkl")

# Load new data for prediction
df = pd.read_csv("Test.csv")  

# Prepare the input features
df["text"] = df["description"].fillna("")
X_new = df[["text", "telecommuting", "has_company_logo", "has_questions"]]

# Predict fraud probability
proba = model.predict_proba(X_new)[:, 1]
df["fake_percentage"] = (proba * 100).round(2)

# Save to CSV
df.to_csv("predicted_jobs.csv", index=False)
print("Predictions saved to predicted_jobs.csv")
