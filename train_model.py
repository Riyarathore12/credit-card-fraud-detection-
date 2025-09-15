import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from joblib import dump

RANDOM_STATE = 42

# Load dataset
df = pd.read_csv("creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# ✅ Oversample fraud using SMOTE
smote = SMOTE(random_state=RANDOM_STATE)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Define models
models = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ]),
    "RandomForest": Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced"
        ))
    ])
}

best_model = None
best_auc = 0
best_name = ""

# Train and evaluate
for name, model in models.items():
    print(f"\n=== Training {name} ===")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred, digits=4))
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    print(f"ROC-AUC: {roc_auc}")
    print(f"PR-AUC : {pr_auc}")

    if roc_auc > best_auc:
        best_auc = roc_auc
        best_model = model
        best_name = name

# Save the best model
bundle = {
    "model_name": best_name,
    "pipeline": best_model,
    "threshold": 0.5
}
dump(bundle, "best_model.joblib")

print(f"\n✅ Saved best model: {best_name} with ROC-AUC {best_auc:.4f}")
