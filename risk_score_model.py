# ---------------------------------------------------
# STEP 6: ACCIDENT RISK CLASSIFICATION MODEL (Improved Accuracy)
# ---------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# ----------------------------
# 1. Load the engineered dataset
# ----------------------------
df = pd.read_csv("processed_data.csv")  # engineered dataset from feature engineering
print(f"✅ Loaded engineered dataset: {df.shape}")

# ----------------------------
# 2. Convert continuous Risk_Score → Risk_Level classes
# ----------------------------
def categorize_risk(score):
    if score < 50:
        return "Low"
    elif score < 80:
        return "Medium"
    else:
        return "High"

# Replace your categorize_risk() block with:
df["Risk_Level"] = df["Risk_Score"].map({
    30: "Low",
    65: "Medium",
    90: "High"
})


# Separate features and target
X = df.drop(columns=["Risk_Score", "Risk_Level"])
y = df["Risk_Level"]

# Load encoders and scaler
try:
    encoders = joblib.load("feature_encoders.pkl")
    scaler = joblib.load("scaler.pkl")
    print("🔄 Loaded encoders and scaler successfully.")
except:
    print("⚠️ Could not load encoders/scaler — proceeding without them.")

# ----------------------------
# 3. Train/Test Split (stratified for balance)
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"🧩 Train/Test Split: {X_train.shape}, {X_test.shape}")

# ----------------------------
# 4. Model Training with Hyperparameter Tuning (Grid Search)
# ----------------------------
from sklearn.model_selection import GridSearchCV

print("🔍 Running GridSearchCV for optimal hyperparameters...")

param_grid = {
    'n_estimators': [200, 400, 600],
    'max_depth': [8, 12, 16],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced']
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid=param_grid,
    cv=3,
    scoring='f1_macro',
    verbose=2,
    n_jobs=-1
)

grid.fit(X_train, y_train)
model = grid.best_estimator_
print(f"🔧 Best Parameters Found: {grid.best_params_}")


# ----------------------------
# 5. Evaluation
# ----------------------------
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")

print("\n📊 Classification Model Performance:")
print(f"✅ Accuracy: {acc:.3f}")
print(f"✅ Macro F1 Score: {f1:.3f}")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=["Low", "Medium", "High"])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Low", "Medium", "High"],
            yticklabels=["Low", "Medium", "High"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Accident Risk Classification")
plt.show()

# ----------------------------
# 6. Feature Importance
# ----------------------------
feat_imp = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(data=feat_imp.head(15), x="Importance", y="Feature")
plt.title("Top 15 Important Features for Accident Risk Classification")
plt.tight_layout()
plt.show()

# ----------------------------
# 7. Save model
# ----------------------------
joblib.dump(model, "risk_model_classifier.pkl")
print("\n✅ Model saved as 'risk_model_classifier.pkl'")

# ----------------------------
# 8. Interpretive Summary
# ----------------------------
print("\n🔍 Interpretive Summary:")
if acc > 0.8:
    print("✅ Excellent model — accurately classifies most accident risks.")
elif acc > 0.6:
    print("🟡 Moderate model — captures key risk patterns but may need more data.")
else:
    print("⚠️ Low accuracy — add better features or tune parameters.")

print("\n💡 Top influential factors driving accident risk:")
print(feat_imp.head(5))
