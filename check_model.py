import pandas as pd
from joblib import load

# Load dataset to confirm column order
df = pd.read_csv("creditcard.csv")
cols = df.drop("Class", axis=1).columns.tolist()  # correct order

print("Column order used during training:", cols)

# Example: take the first fraud transaction directly from dataset
fraud_row = df[df["Class"] == 1].iloc[0].drop("Class")
legit_row = df[df["Class"] == 0].iloc[0].drop("Class")

fraud_df = fraud_row.to_frame().T
legit_df = legit_row.to_frame().T

# Load trained model
bundle = load("best_model.joblib")
pipeline = bundle["pipeline"]

print("\nModel:", bundle["model_name"])
print("Threshold:", bundle["threshold"])

# Legit transaction
print("\n--- Legit transaction ---")
print("Proba:", pipeline.predict_proba(legit_df)[:, 1])
print("Prediction:", pipeline.predict(legit_df))

# Fraud transaction
print("\n--- Fraud transaction ---")
print("Proba:", pipeline.predict_proba(fraud_df)[:, 1])
print("Prediction:", pipeline.predict(fraud_df))
